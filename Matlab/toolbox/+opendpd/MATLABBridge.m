classdef MATLABBridge < handle
    %MATLABBRIDGE Execute Studio MATLINK's named operations inside MATLAB.
    % A timer publishes variable metadata and processes transfer requests.
    % Browser content never becomes a MATLAB expression or arbitrary command.
    properties (SetAccess = private)
        Project
        ClientID (1,1) string = ""
        Connected = false
        LastError (1,1) string = ""
    end
    properties (Access = private)
        Backend = []
        PollTimer = []
        OwnsConnection = false
        Closing = false
        Busy = false
        Outcomes
        OutcomeOrder = {}
        Imported
    end
    methods
        function obj = MATLABBridge(project, options)
            arguments
                project (1,1) opendpd.Project
                options.OwnsConnection (1,1) logical = false
                options.Label (1,1) string = "MATLAB R" + string(version('-release'))
            end
            obj.Project = project;
            obj.OwnsConnection = options.OwnsConnection;
            obj.Outcomes = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.Imported = containers.Map('KeyType', 'char', 'ValueType', 'char');
            module = py.importlib.import_module('opendpd.sdk.matlink');
            obj.Backend = module.MatlinkClient(project.Backend, char(options.Label), ...
                version('-release'), obj.variablesJSON(), jsonencode({'dataset_export', 'result_bundle'}));
            obj.ClientID = string(obj.Backend.client_id);
            obj.Connected = true;
            obj.PollTimer = timer('Name', 'OpenDPD MATLINK', 'ExecutionMode', 'fixedSpacing', ...
                'StartDelay', 1, 'Period', 2, 'BusyMode', 'drop', 'TimerFcn', @(~,~) obj.poll());
            start(obj.PollTimer);
        end

        function poll(obj)
            %POLL Publish presence, execute pending requests and acknowledge once.
            if obj.Closing || obj.Busy, return; end
            obj.Busy = true;
            finish = onCleanup(@() obj.finishPoll()); %#ok<NASGU>
            try
                response = jsondecode(char(obj.Backend.heartbeat(obj.variablesJSON())));
                obj.Connected = true;
                obj.LastError = "";
                queued = {};
                for index = 1:numel(response.requests)
                    if iscell(response.requests), item = response.requests{index};
                    else, item = response.requests(index); end
                    queued{end+1} = char(item.request_id); %#ok<AGROW>
                end
                % A lost acknowledgment must not run the MATLAB operation twice.
                cached = keys(obj.Outcomes);
                attempted = {};
                for index = 1:numel(cached)
                    entry = obj.Outcomes(cached{index});
                    if ~entry.acknowledged
                        if ~any(strcmp(queued, cached{index}))
                            % The broker's complete queued set confirms that a
                            % lost acknowledgment was stored or the request retired.
                            entry.acknowledged = true;
                            obj.Outcomes(cached{index}) = entry;
                            continue
                        end
                        attempted{end+1} = cached{index}; %#ok<AGROW>
                        obj.tryAcknowledge(cached{index});
                    end
                end
                for index = 1:min(numel(response.requests), 4)
                    if iscell(response.requests), request = response.requests{index};
                    else, request = response.requests(index); end
                    key = char(request.request_id);
                    if any(strcmp(attempted, key)), continue; end
                    if ~isKey(obj.Outcomes, key)
                        try
                            result = obj.execute(request);
                            outcome = struct('status', 'succeeded', 'result', result);
                        catch cause
                            message = cause.message;
                            if numel(message) > 2000, message = [message(1:1997) '...']; end
                            outcome = struct('status', 'failed', 'error', message);
                        end
                        obj.Outcomes(key) = struct('outcome', outcome, 'acknowledged', false);
                        obj.OutcomeOrder{end+1} = key;
                    end
                    obj.tryAcknowledge(key);
                end
                while numel(obj.OutcomeOrder) > 256
                    removable = find(cellfun(@(key) obj.Outcomes(key).acknowledged, obj.OutcomeOrder), 1);
                    if isempty(removable), break; end
                    oldest = obj.OutcomeOrder{removable};
                    remove(obj.Outcomes, oldest);
                    obj.OutcomeOrder(removable) = [];
                end
            catch cause
                obj.Connected = false;
                obj.LastError = string(cause.message);
            end
        end

        function current = isCurrent(obj)
            %ISCURRENT Whether this connection still belongs to the live service.
            current = false;
            if obj.Closing || isempty(obj.Backend), return; end
            try, current = logical(obj.Backend.is_current()); catch, end
        end

        function delete(obj)
            if obj.Closing, return; end
            obj.Closing = true;
            if ~isempty(obj.PollTimer) && isvalid(obj.PollTimer)
                stop(obj.PollTimer); delete(obj.PollTimer);
            end
            obj.PollTimer = [];
            if ~isempty(obj.Backend)
                try, obj.Backend.disconnect(); catch, end
            end
            if ~isempty(obj.Project)
                matlinkRegistry('remove', char(obj.Project.Workspace), obj);
                if obj.OwnsConnection
                    try, opendpd.closeProject(obj.Project); catch, end
                end
            end
            obj.Connected = false;
        end
    end
    methods (Access = private)
        function tryAcknowledge(obj, key)
            try
                obj.acknowledge(key);
            catch cause
                % Preserve the outcome for another attempt, without stopping
                % heartbeats or executing the underlying operation again.
                obj.LastError = string(cause.message);
            end
        end

        function acknowledge(obj, key)
            entry = obj.Outcomes(key);
            obj.Backend.complete(key, jsonencode(entry.outcome));
            entry.acknowledged = true;
            obj.Outcomes(key) = entry;
        end

        function finishPoll(obj)
            obj.Busy = false;
        end

        function encoded = variablesJSON(obj) %#ok<MANU>
            items = evalin('base', 'whos');
            records = {};
            for index = 1:numel(items)
                item = items(index);
                if isempty(regexp(item.name, '^[A-Za-z][A-Za-z0-9_]{0,62}$', 'once')) ...
                        || ~any(strcmp(item.class, {'single', 'double', 'struct'})) ...
                        || numel(item.size) > 32 || prod(item.size) > 2^53
                    continue
                end
                eligible = any(strcmp(item.class, {'single', 'double'})) && ~item.sparse ...
                    && numel(item.size) == 2 && any(item.size == 1) && prod(item.size) > 0;
                records{end+1} = struct('name', item.name, 'class_name', item.class, ... %#ok<AGROW>
                    'size', double(item.size), 'complex', logical(item.complex), ...
                    'eligible', eligible, 'n_samples', double(prod(item.size)));
                if numel(records) == 256, break; end
            end
            encoded = jsonencode(records);
        end

        function result = execute(obj, request)
            payload = request.payload;
            switch string(request.action)
                case "create_demo"
                    result = obj.createDemo();
                case "import_iq"
                    xName = obj.identifier(obj.field(payload, 'input', ''));
                    yName = obj.identifier(obj.field(payload, 'output', ''));
                    if strcmp(xName, yName)
                        error('opendpd:SignalPair', 'Choose different PA input and output variables.');
                    end
                    x = obj.baseVariable(xName);
                    y = obj.baseVariable(yName);
                    sampleRate = obj.positive(payload, 'sample_rate_mhz') * 1e6;
                    bandwidth = obj.positive(payload, 'bandwidth_mhz') * 1e6;
                    segment = obj.positive(payload, 'segment_samples');
                    name = obj.field(payload, 'name', '');
                    dataset = opendpd.importIQ(obj.Project, x, y, SampleRate=sampleRate, ...
                        Bandwidth=bandwidth, SegmentSamples=segment, Name=string(name), ...
                        Origin=string(obj.field(payload, 'origin', 'unknown')));
                    result = struct('dataset_id', dataset.dataset_id, 'display_name', dataset.display_name, ...
                        'n_samples', dataset.n_samples);
                case "import_result"
                    result = obj.importReport(payload);
                case "import_dataset"
                    result = obj.importDataset(obj.field(payload, 'dataset_id', ''));
                case "open_variable"
                    name = obj.identifier(obj.field(payload, 'variable', ''));
                    items = evalin('base', 'whos');
                    index = find(strcmp({items.name}, name), 1);
                    if isempty(index)
                        error('opendpd:VariableMissing', 'The variable was cleared. Send the report to MATLAB again.');
                    end
                    if ~any(strcmp(items(index).class, {'single', 'double'})) && ~any(strcmp(values(obj.Imported), name))
                        error('opendpd:VariableType', 'Open a numeric signal or a report imported by this MATLINK connection.');
                    end
                    openvar(name);
                    result = struct('variable', name);
                otherwise
                    error('opendpd:MATLINKAction', 'Unsupported MATLINK operation.');
            end
        end

        function result = importReport(obj, payload)
            runID = obj.field(payload, 'run_id', '');
            requested = obj.field(payload, 'variable', '');
            bundle = isfield(payload, 'bundle') && isequal(payload.bundle, true);
            if ~isempty(requested), obj.identifier(requested); end
            cacheKey = [runID ':' requested ':' num2str(bundle)];
            job = opendpd.getRun(obj.Project, string(runID));
            record = opendpd.status(job);
            if ~strcmp(record.status, 'succeeded')
                error('opendpd:ResultUnavailable', 'This experiment has not completed successfully.');
            end
            names = evalin('base', 'who');
            if isKey(obj.Imported, cacheKey) && any(strcmp(names, obj.Imported(cacheKey)))
                current = evalin('base', obj.Imported(cacheKey));
                if isstruct(current) && isscalar(current) && isfield(current, 'run_id') ...
                        && (ischar(current.run_id) || isstring(current.run_id)) && strcmp(current.run_id, runID)
                    result = struct('variable', obj.Imported(cacheKey), 'run_id', runID);
                    return
                end
            end
            if bundle
                % MATLAB caches Python member names across importlib.reload.
                % getattr also supports upgrading an existing desktop session.
                exportReport = py.getattr(obj.Backend, 'result_bundle');
                report = jsondecode(char(exportReport(runID)));
            else, report = opendpd.result(job); end
            tags = struct('train_pa', 'pa', 'train_dpd', 'dpd', 'run_dpd', 'dpdExport', ...
                'evaluate_pa', 'paEvaluation', 'evaluate_measured', 'measured');
            tag = 'result';
            if isfield(tags, record.task), tag = tags.(record.task); end
            parts = string({tag, obj.field(record, 'model_key', ''), obj.field(record, 'dataset_id', '')});
            base = matlab.lang.makeValidName(char(strjoin(parts(strlength(parts) > 0), '_')));
            if ~isempty(requested), base = requested; end
            variable = matlab.lang.makeUniqueStrings(base, evalin('base', 'who'), namelengthmax);
            assignin('base', variable, report);
            obj.Imported(cacheKey) = variable;
            result = struct('variable', variable, 'run_id', runID);
        end

        function result = importDataset(obj, datasetID)
            cacheKey = ['dataset:' datasetID];
            if isKey(obj.Imported, cacheKey) && any(strcmp(evalin('base', 'who'), obj.Imported(cacheKey)))
                current = evalin('base', obj.Imported(cacheKey));
                if isstruct(current) && isscalar(current) && isfield(current, 'dataset_id') ...
                        && isequal(current.dataset_id, datasetID)
                    result = struct('variable', obj.Imported(cacheKey), 'dataset_id', datasetID);
                    return
                end
            end
            exportDataset = py.getattr(obj.Backend, 'dataset');
            exported = exportDataset(datasetID);
            signal = jsondecode(char(exported{1}));
            captures = exported{2};
            paired = repmat(struct('dataset_id', '', 'label', '', 'x', [], 'y', [], ...
                'signal', struct(), 'split', struct(), 'simulation', struct()), double(py.len(captures)), 1);
            for index = 1:double(py.len(captures))
                item = captures{index};
                metadata = jsondecode(char(item{1}));
                x = single(item{2}); y = single(item{3});
                paired(index) = struct('dataset_id', metadata.dataset_id, ...
                    'label', metadata.display_name, 'x', complex(x(:,1), x(:,2)), ...
                    'y', complex(y(:,1), y(:,2)), 'signal', metadata.signal, ...
                    'split', metadata.split, 'simulation', metadata.simulation); %#ok<AGROW>
            end
            signal.captures = paired;
            variable = matlab.lang.makeUniqueStrings('opendpdSignals', evalin('base', 'who'), namelengthmax);
            assignin('base', variable, signal);
            obj.Imported(cacheKey) = variable;
            result = struct('variable', variable, 'dataset_id', datasetID, 'n_captures', numel(paired));
        end

        function result = createDemo(obj) %#ok<MANU>
            names = evalin('base', 'who');
            inputName = matlab.lang.makeUniqueStrings('opendpdDemoInput', names);
            outputName = matlab.lang.makeUniqueStrings('opendpdDemoOutput', [names; {inputName}]);
            stream = RandStream('mt19937ar', 'Seed', 42);
            n = 4096; fs = 80e6; bw = 20e6;
            f = (-n/2:n/2-1).' * fs/n;
            spectrum = complex(randn(stream,n,1), randn(stream,n,1));
            spectrum(abs(f) > bw/2) = 0;
            x = ifft(ifftshift(spectrum));
            x = single(0.2*x/sqrt(mean(abs(x).^2)));
            y = x - 0.25*x.*abs(x).^2 + 0.03*[complex(single(0)); x(1:end-1)];
            assignin('base', inputName, x);
            assignin('base', outputName, y);
            result = struct('input', inputName, 'output', outputName, 'sample_rate_mhz', 80, ...
                'bandwidth_mhz', 20, 'segment_samples', 256, 'origin', 'synthetic', 'n_samples', n, ...
                'name', ['matlab-demo-' char(datetime('now', 'Format', 'yyyyMMdd-HHmmssSSS'))]);
        end
    end
    methods (Static, Access = private)
        function value = baseVariable(name)
            % evalin on a name that is not a workspace variable would call a function of that name, so the
            % variable must exist now, not only in the last heartbeat that the browser saw.
            if ~ismember(name, evalin('base', 'who'))
                error('opendpd:VariableMissing', ...
                    'Variable %s is no longer in the MATLAB base workspace. Refresh MATLINK and select it again.', name);
            end
            value = evalin('base', name);
        end

        function value = field(payload, name, fallback)
            if ~isfield(payload, name) || isempty(payload.(name)), value = fallback; return; end
            value = payload.(name);
            if ~(ischar(value) && isrow(value) || isstring(value) && isscalar(value))
                error('opendpd:MATLINKField', '%s must be text.', name);
            end
            value = char(value);
        end

        function name = identifier(value)
            name = char(value);
            if ~isvarname(name)
                error('opendpd:VariableName', 'Choose a single MATLAB variable name, without expressions.');
            end
        end

        function value = positive(payload, name)
            if ~isfield(payload, name), error('opendpd:SignalMetadata', 'Provide sample rate and bandwidth.'); end
            value = payload.(name);
            if ~isnumeric(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
                error('opendpd:SignalMetadata', '%s must be positive and finite.', name);
            end
        end
    end
end
