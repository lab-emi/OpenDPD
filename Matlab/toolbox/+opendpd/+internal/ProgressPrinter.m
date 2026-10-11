classdef ProgressPrinter < handle
    % Prints the events the Python side appends to progress.jsonl (one JSON object per line) while fit runs: progress
    % lines when Verbose, and always a warning if the service fit started could not be stopped afterwards.
    properties (Access = private)
        File
        Verbose
        Offset = 0
        Models
        LastEpoch = struct('train_pa', -1, 'train_dpd', -1)
    end
    methods
        function obj = ProgressPrinter(file, verbose, dpdModel, paModel)
            obj.File = file;
            obj.Verbose = verbose;
            obj.Models = struct('train_pa', paModel, 'train_dpd', dpdModel);
        end

        function poll(obj)
            if ~isfile(obj.File)
                return
            end
            fid = fopen(obj.File, 'r');
            closeFile = onCleanup(@() fclose(fid));
            fseek(fid, obj.Offset, 'bof');
            text = fread(fid, Inf, '*char').';
            lastNewline = find(text == newline, 1, 'last');
            if isempty(lastNewline)
                return
            end
            obj.Offset = obj.Offset + lastNewline;                 % only whole lines are consumed
            for line = splitlines(string(text(1:lastNewline))).'
                if strlength(strtrim(line)) == 0
                    continue
                end
                try
                    event = jsondecode(char(line));
                catch
                    continue                                       % a progress line is never worth stopping a training run for
                end
                obj.show(event);
            end
        end
    end
    methods (Access = private)
        function show(obj, event)
            if strcmp(event.stage, 'close')
                warning('opendpd:fit:ServiceNotStopped', ['The training finished, but the Studio service that opendpd.fit ' ...
                    'started has not stopped (%s). It holds the workspace until it does.'], event.warning);
                return
            end
            if ~obj.Verbose
                return
            end
            switch event.stage
                case 'import'
                    fprintf('opendpd.fit: importing the capture\n');
                case {'train_pa', 'train_dpd'}
                    label = 'PA';
                    if strcmp(event.stage, 'train_dpd')
                        label = 'DPD';
                    end
                    epoch = [];
                    if isfield(event, 'epoch') && ~isempty(event.epoch)
                        epoch = event.epoch;
                    end
                    total = [];
                    if isfield(event, 'total_epochs') && ~isempty(event.total_epochs)
                        total = event.total_epochs;
                    end
                    if isempty(epoch)
                        fprintf('opendpd.fit: %s (%s) %s, run %s\n', label, obj.Models.(event.stage), event.status, event.run_id);
                    elseif ~isempty(total) && epoch ~= obj.LastEpoch.(event.stage) && ...
                            (epoch == total || mod(epoch, max(1, ceil(total / 10))) == 0)
                        obj.LastEpoch.(event.stage) = epoch;
                        fprintf('opendpd.fit: %s (%s) epoch %d of %d (%s)\n', label, obj.Models.(event.stage), epoch, total, event.status);
                    end
                case 'export'
                    fprintf('opendpd.fit: exporting the %s model package\n', upper(event.role));
            end
        end
    end
end
