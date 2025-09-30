function RS_SMW200A_DPD_Test(Set_Parent_Flag, F_awg, In_Amp_1, I_data, Q_data, filename)
%__author__ = "Yizhuo Wu"
%__license__ = "Apache-2.0 License"
%__email__ = "yizhuo.wu@tudelft.nl"

% Function to control R&S SMW200A for DPD testing
% Parameters similar to Key_AWG8901A_DPD_Test for compatibility
% 
% Inputs:
%   Set_Parent_Flag: Initial setup flag (1 for first configuration)
%   F_awg: Sampling frequency in Hz
%   In_Amp_1: Channel 1 amplitude scaling
%   I_data: I component of the signal
%   Q_data: Q component of the signal
%   filename: Waveform filename

% Create VISA object for SMW200A
% Note: Update the VISA address according to your setup
% visaobj = visadev('GPIB1::28::INSTR');
RS_SMW = visa('agilent','TCPIP0::169.254.2.22::inst0::INSTR');
fopen(RS_SMW)
visaobj = RS_SMW;
instrument.visa_handle = visaobj;
instrument.wvfile = @(filepath) fprintf(visaobj, 'MMEM:LOAD:STAT 1,"%s"\n', filepath);

% Create waveform generator instance
wvgen = RSWaveformGenerator(instrument);
wvgen.instrument_filename = filename;

% Generate I/Q signal parameters
clock = F_awg;           % Sample rate
duration = 0.001;        % Signal duration: 1 ms

% Initial setup if Set_Parent_Flag is true
if(Set_Parent_Flag == 0)
    % Query instrument ID
    idn = query(RS_SMW, '*IDN?');
    disp(idn);
    
    % Turn off RF outputs initially
    fprintf(RS_SMW, ':OUTP1:STAT OFF');
    
    % Stop current output
    fprintf(RS_SMW, ':SOUR1:BB:ARB:STAT OFF');

    amp1_cmd = sprintf(':SOUR1:POW:LEV:IMM:AMPL %f', In_Amp_1);
    fprintf(RS_SMW, amp1_cmd);
        
    % Query actual amplitude
    actual_amp1 = query(RS_SMW, ':SOUR1:POW:LEV:IMM:AMPL?');
    disp(['Channel 1 amplitude: ' actual_amp1]);
    
    
    % Create markers
    % Note: In MATLAB, use matrix with semicolons for rows
    markers = struct();
    markers.marker1 = [0,0; 10,1; 50,0];  % First marker pattern
    markers.marker2 = [0,1];              % Second marker pattern
    
    % Generate the waveform
    fprintf('Generating waveform...\n');
    wvgen.generate_wave(I_data, Q_data, clock, markers);
    
    % Save waveform locally (optional)
    fprintf('Saving waveform locally...\n');
    wvgen.save_wave_file(filename);
    fclose(RS_SMW)
    % Upload to instrument using FTP and SCPI commands
    fprintf('Uploading waveform to instrument...\n');
    wvgen.upload_wave('169.254.2.22', 'instrument', 'instrument', '/user');
    fopen(RS_SMW)
    % Configure instrument
    fprintf('Configuring instrument...\n');


elseif(Set_Parent_Flag == 1)
    
    % Configure basic ARB settings
    fprintf(RS_SMW, ':OUTP1:STAT ON');
    fprintf(RS_SMW, ':SOUR1:BB:ARB:STAT ON');
    fprintf(RS_SMW, ':SOUR1:CORR:OPT:RF:CHAR EVM');
    
end

% Close VISA connection
fclose(RS_SMW);
clear RS_SMW;
end
