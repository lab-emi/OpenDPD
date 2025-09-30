%__author__ = "Yizhuo Wu, Qian Wu"
%__license__ = "Apache-2.0 License"
%__email__ = "yizhuo.wu@tudelft.nl,qian.wu@ucdconnect.ie"

classdef RSWaveformGenerator < handle
    % Generates .wv files from I/Q for the AFQ 100B I/Q modulation generator
    % and related Rohde & Schwarz instruments
    %
    % RSWaveformGenerator(instrument)
    %     Initialises waveform generator ready to upload to instrument
    %     If instrument is not provided, can generate and save .wv files
    %         but not upload to instrument
    
    properties
        instrument
        instrument_directory = 'var\user'
        instrument_filename = 'temp.wv'
        instrument_filepath = 'var\user\temp.wv'
        comment = ''
        copyright = ''
        normalise = false
        checks = false
        waveform
        max_samples = 100e6 % 512e6 Device memory 512MSa but generating long waveform is too memory intensive
    end
    
    methods
        function obj = RSWaveformGenerator(instrument)
            % Constructor
            if nargin > 0
                obj.instrument = instrument;
            end
        end
        
        function generate_wave(obj, I_data, Q_data, clock, markers)
            % Generates waveform (R&S .wv format) from input data
            % I_data must be vector of values in range (-1,1)
            % Q_data must be vector of values in range (-1,1)
            % clock is AWG sample rate, from 1kHz to 300MHz, or 600MHz
            % markers is struct with fields 'marker1','marker2','marker3','marker4'
            
            % Sanity checks
            obj.waveform = [];
            I_data_len = length(I_data);
            Q_data_len = length(Q_data);
            
            if I_data_len > obj.max_samples
                error('Number of samples %d exceeds max_samples %d', I_data_len, obj.max_samples);
            end
            if I_data_len ~= Q_data_len
                error('I_data and Q_data are not same length (%d,%d)', I_data_len, Q_data_len);
            end
            
            % Convert to single precision
            I_data = single(I_data);
            Q_data = single(Q_data);
            
            % Format I,Q vectors into IQIQIQ...
            IQ_data_len = 2 * I_data_len;
            IQ_data = zeros(1, IQ_data_len, 'single');
            IQ_data(1:2:end) = I_data;
            IQ_data(2:2:end) = Q_data;
            
            % If scaling is desired, normalise to peak vector length of 1.0
            if obj.normalise
                max_IQ_data = max(abs(I_data + 1i*Q_data));
                IQ_data = IQ_data / max_IQ_data;
                peak = 1.0;
                max_IQ_data = 1.0;
                rms = sqrt(mean(IQ_data(1:2:end).^2 + IQ_data(2:2:end).^2)) / max_IQ_data;
                crf = 20*log10(peak/rms); % Crest factor
            else
                % If not scaling, check for clipping if enabled
                if obj.checks
                    if max(I_data) > 1.0 || min(I_data) < -1.0
                        error('I_data must be in range -1 to +1 if auto scaling is disabled.');
                    end
                    if max(Q_data) > 1.0 || min(Q_data) < -1.0
                        error('Q_data must be in range -1 to +1 if auto scaling is disabled.');
                    end
                    if max(abs(I_data + 1i*Q_data)) > 1.0
                        error('I/Q vector length must be <1 if auto scaling is disabled.');
                    end
                end
                peak = 1.0;
                rms = 1.0;
                crf = 0.0;
            end
            
            % Convert IQ_data to int16
            IQ_data = int16(floor(IQ_data*32767 + 0.5));
            
            % Generate wv file header
            header_tag_str = '{TYPE: SMU-WV, 0}';
            if ~isempty(obj.comment)
                comment_str = sprintf('{COMMENT: %s}', obj.comment);
            else
                comment_str = '';
            end
            if ~isempty(obj.copyright)
                copyright_str = sprintf('{COPYRIGHT: %s}', obj.copyright);
            else
                copyright_str = '';
            end
            origin_info_str = '{ORIGIN INFO: MATLAB}';
            level_offs_str = sprintf('{LEVEL OFFS: %f, %f}', 20*log10(1.0/rms), 20*log10(1.0/peak));
            
            % Get current date and time
            current_datetime = datetime('now');
            date_str = sprintf('{DATE: %s;%s}', datestr(current_datetime, 'yyyy-mm-dd'), datestr(current_datetime, 'HH:MM:SS'));
            clock_str = sprintf('{CLOCK: %f}', clock);
            samples_str = sprintf('{SAMPLES: %d}', I_data_len);
            
            waveform_header = uint8([header_tag_str, comment_str, copyright_str, origin_info_str, ...
                level_offs_str, date_str, clock_str, samples_str]);
            
            % Generate markers
            waveform_markers = '';
            if nargin > 4 && ~isempty(markers)
                if ~isstruct(markers)
                    error('Markers must be struct with fields marker1, marker2, marker3, marker4');
                end
                waveform_markers = sprintf('{CONTROL LENGTH: %d}', IQ_data_len);
                marker_fields = {'marker1', 'marker2', 'marker3', 'marker4'};
                for i = 1:length(marker_fields)
                    if isfield(markers, marker_fields{i})
                        waveform_markers = [waveform_markers, sprintf('{MARKER LIST %d: %s}', ...
                            i, obj.generate_marker_string(markers.(marker_fields{i})))];
                    end
                end
            end
            waveform_markers = uint8(waveform_markers);
            
            % Convert IQ_data to binary
            wv_file_IQ_data = uint8([sprintf('{WAVEFORM-%d: #', 2*IQ_data_len + 1), typecast(IQ_data, 'uint8'), '}']);
            
            % Combine all parts
            obj.waveform = [waveform_header, waveform_markers, wv_file_IQ_data];
            
            fprintf('Waveform generated: %d samples, %d bytes\n', I_data_len, length(obj.waveform));
        end
        
        function upload_wave(obj, ftp_host, ftp_username, ftp_password, ftp_remote_dir)
            % Upload waveform using FTP and SCPI commands (Python equivalent method)
            % Parameters:
            %   ftp_host - FTP server IP address
            %   ftp_username - FTP username (default: 'instrument')
            %   ftp_password - FTP password (default: 'instrument')  
            %   ftp_remote_dir - Remote directory (default: '/user')
            
            if isempty(obj.waveform)
                error('Waveform not generated. Please run generate_wave() or read_wave_file()');
            end
            
            % Set default parameters
            if nargin < 2
                error('FTP host must be provided');
            end
            if nargin < 3
                ftp_username = 'instrument';
            end
            if nargin < 4
                ftp_password = 'instrument';
            end
            if nargin < 5
                ftp_remote_dir = '/user';
            end
            
            % First save waveform to local file
            local_filename = obj.instrument_filename;
            obj.save_wave_file(local_filename);
            
            % --- FTP Upload ---
            try
                fprintf('Uploading %s via FTP...', local_filename);
                
                % Ensure host is string
                host = ftp_host;
                if isempty(host)
                    error('FTP host is empty');
                end
                
                % Create FTP connection
                ftp_obj = ftp(host, ftp_username, ftp_password);
                
                % Change to remote directory
                if ~isempty(ftp_remote_dir)
                    cd(ftp_obj, ftp_remote_dir);
                end
                
                % Upload file
                mput(ftp_obj, local_filename);
                
                % Close FTP connection
                close(ftp_obj);
                fprintf(' FTP upload completed\n');
                
            catch ME
                fprintf(' FTP upload failed: %s\n', ME.message);
                if exist('ftp_obj', 'var')
                    try
                        close(ftp_obj);
                    catch
                        % Ignore close errors
                    end
                end
                return;
            end
            
            % --- SCPI Command to Load Waveform ---
            try
                % Generate SCPI command (equivalent to Python)
                scpi_cmd = sprintf('SOURce:BB:ARBitrary:WAVeform:SELect "/var/user/%s"', obj.instrument_filename);
                
                % Use VISA to send SCPI command
                instr = visa('agilent',sprintf('TCPIP0::%s::inst0::INSTR', ftp_host));
                fopen(instr)
                fprintf(instr, scpi_cmd);
                fclose(instr)
                clear instr;
                fprintf('Waveform loaded on instrument\n');
                
            catch ME
                fprintf('SCPI command failed: %s\n', ME.message);
            end
            
            fprintf('Upload completed: %s\n', obj.instrument_filename);
        end
        
        function save_wave_file(obj, local_filename)
            if isempty(obj.waveform)
                error('Waveform not generated. Please run generate_wave() or read_wave_file()');
            end
            fileID = fopen(local_filename, 'w');
            fwrite(fileID, obj.waveform);
            fclose(fileID);
        end
        
        function read_wave_file(obj, local_filename)
            fileID = fopen(local_filename, 'r');
            obj.waveform = fread(fileID, inf, 'uint8');
            fclose(fileID);
        end
        
        function upload_wave_file(obj, local_filename)
            obj.read_wave_file(local_filename);
            obj.upload_wave();
        end
    end
    
    methods (Access = private)
        function marker_string = generate_marker_string(obj, marker_array)
            % Convert marker array to string format
            if size(marker_array, 2) ~= 2
                error('Marker array must be in format [0,0;20,1;50,0], even if one entry');
            end
            
            marker_string = '';
            for i = 1:size(marker_array, 1)
                if i > 1
                    marker_string = [marker_string, ';'];
                end
                marker_string = [marker_string, sprintf('%d:%d', marker_array(i,1), marker_array(i,2))];
            end
        end
    end
end
