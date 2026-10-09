classdef Session < handle & matlab.mixin.CustomDisplay
    % OPENDPD.LAB.SESSION A supervised measurement session: RF stays off until a named person arms it, every limit is
    % checked before anything is sent, and every abnormal path switches RF off and leaves the session tripped.
    %
    %   lab = opendpd.lab.Session(MeasureFcn=@myMeasure, RFOffFcn=@myRFOff, Operator="Your Name", ...
    %       MaxPeak=0.9, MaxOutputPower_dBm=30, PowerCalibration=@myPower, Timeout=60);
    %   arm(lab);                                    % a real chain also needs OPENDPD_ALLOW_RF_OUTPUT=1 (see below)
    %   y = measure(lab, u, SampleRate=fs);          % checks the limits, then calls myMeasure(u, fs)
    %   disarm(lab);
    %   ds = opendpd.importIQ(p, u, y, SampleRate=fs, Bandwidth=bw, SegmentSamples=2048, Origin="measured", ...
    %       Source=record(lab, Compact=true));
    %   saveRecord(lab, "session-1.json");           % the complete log, next to your data
    %
    % Your functions are the instrument adapter; nothing else is:
    %   MeasureFcn(u, fs)   plays the complex baseband signal u (digital full scale 1.0) at sample rate fs and returns
    %                       the captured complex vector (or N-by-2 [I Q]). Called with MeasureFcn(u) when measure is
    %                       called without SampleRate.
    %   RFOffFcn()          switches the output off. Required, with no default: a session that cannot switch RF off is
    %                       not a safe session. It must be idempotent and must not depend on what MeasureFcn left half done.
    %   HeartbeatFcn()      optional; returns when the link to the instrument is alive and errors otherwise.
    %   PowerCalibration(u) optional; returns the output power in dBm that playing u produces, as a finite number. Whether
    %                       that is average or peak power is your call: for a safety ceiling use peak.
    %
    % A session built from functions is treated as able to emit RF: arming it needs the operator's name AND the environment
    % variable OPENDPD_ALLOW_RF_OUTPUT=1, which only an approved laboratory session sets. This toolbox, its tests and
    % automated tooling never set it. A session on an opendpd.lab.MockInstrument emits nothing and arms with a name alone.
    % This is the same rule as the Python interlock (opendpd/instruments/safety.py, docs/architecture/instruments.md).
    %
    % States: disarmed -> armed -> active (inside measure) -> armed -> disarmed. Anything abnormal - an error in your
    % functions, a timeout, a lost link, a non-finite or empty capture, an interrupt (Ctrl+C), abort, a failed RF-off -
    % switches RF off and leaves the session "tripped". A tripped session cannot be armed again: make a new one.
    % Deleting a session that is still armed switches RF off. After every measurement RF is switched off again unless
    % RFOffAfterMeasure=false.
    %
    % Stricter than the Python interlock: if MaxOutputPower_dBm is set, a measurement whose output power is not known (no
    % PowerCalibration and no RequestedPower_dBm) is refused, so a ceiling can never silently not apply.
    %
    % What MATLAB cannot do: interrupt a running function. Timeout is checked when MeasureFcn returns, so give your
    % instrument calls their own timeouts (for example the Timeout property of a visadev) and let a hung instrument
    % return by itself; the heartbeat is checked before and after a measurement, not during it. Ctrl+C trips the session
    % through an onCleanup guard (MATLAB documents that onCleanup runs on Ctrl+C); that path could not be automated in tests.
    properties (SetAccess = private)
        Name (1,1) string = ""
        Description (1,1) string = ""
        Operator (1,1) string = ""
        State (1,1) string = "disarmed"
        TripReason (1,1) string = ""
        RFOffFailed (1,1) logical = false
        IsMock (1,1) logical = false
        MaxPeak (1,1) double = 1
        MaxOutputPower_dBm (1,1) double = NaN
        Timeout (1,1) double = 30
        RFOffAfterMeasure (1,1) logical = true
        Log = repmat(struct('At', '', 'Event', '', 'State', '', 'Detail', struct()), 0, 1)
        Measurements = repmat(struct('index', 0, 'at', '', 'n_played', 0, 'n_captured', 0, 'sample_rate_hz', NaN, ...
            'peak', 0, 'power_dbm', NaN, 'played_sha256', '', 'captured_sha256', '', 'seconds', 0), 0, 1)
    end
    properties (Access = private)
        MeasureFcn
        RFOffFcn
        HeartbeatFcn
        PowerCalibration
        StartedAt (1,1) string = ""
        PendingReason (1,1) string = ""
    end
    properties (Constant)
        Gate = "OPENDPD_ALLOW_RF_OUTPUT"
        Format = "opendpd-lab-session-v1"
    end

    methods
        function obj = Session(options)
            arguments
                options.MeasureFcn = []
                options.RFOffFcn = []
                options.HeartbeatFcn = []
                options.PowerCalibration = []
                options.Instrument = []
                options.Operator (1,1) string = ""
                options.Name (1,1) string = ""
                options.Description (1,1) string = ""
                options.MaxPeak (1,1) double {mustBePositive, mustBeFinite} = 1
                options.MaxOutputPower_dBm (1,1) double {mustBeReal} = NaN
                options.Timeout (1,1) double {mustBePositive, mustBeFinite} = 30
                options.RFOffAfterMeasure (1,1) logical = true
            end
            if ~isempty(options.Instrument)
                inst = options.Instrument;
                if ~strcmp(class(inst), 'opendpd.lab.MockInstrument')
                    error('opendpd:lab:Instrument', ['Instrument must be an opendpd.lab.MockInstrument. A real chain is ' ...
                        'described by MeasureFcn and RFOffFcn, so that it is treated as able to emit RF.']);
                end
                if ~isempty(options.MeasureFcn) || ~isempty(options.RFOffFcn) || ~isempty(options.HeartbeatFcn)
                    error('opendpd:lab:Instrument', 'Give either Instrument or MeasureFcn/RFOffFcn/HeartbeatFcn, not both.');
                end
                options.MeasureFcn = @(varargin) inst.measure(varargin{:});
                options.RFOffFcn = @() inst.rfOff();
                options.HeartbeatFcn = @() inst.heartbeat();
                obj.IsMock = true;
                info = inst.describe();
                if strlength(options.Name) == 0, options.Name = info.name; end
                if strlength(options.Description) == 0, options.Description = info.description; end
            end
            if ~isa(options.MeasureFcn, 'function_handle') || ~isa(options.RFOffFcn, 'function_handle')
                error('opendpd:lab:Instrument', ['A session needs MeasureFcn and RFOffFcn (function handles). There is no ' ...
                    'default RFOffFcn: a session that cannot switch RF off is not a safe session.']);
            end
            for field = ["HeartbeatFcn", "PowerCalibration"]
                if ~isempty(options.(field)) && ~isa(options.(field), 'function_handle')
                    error('opendpd:lab:Instrument', '%s must be a function handle.', field);
                end
            end
            obj.MeasureFcn = options.MeasureFcn;
            obj.RFOffFcn = options.RFOffFcn;
            obj.HeartbeatFcn = options.HeartbeatFcn;
            obj.PowerCalibration = options.PowerCalibration;
            obj.Operator = strtrim(options.Operator);
            obj.Name = options.Name;
            obj.Description = options.Description;
            obj.MaxPeak = options.MaxPeak;
            obj.MaxOutputPower_dBm = options.MaxOutputPower_dBm;
            obj.Timeout = options.Timeout;
            obj.RFOffAfterMeasure = options.RFOffAfterMeasure;
            obj.StartedAt = opendpd.lab.Session.timestamp();
            obj.log_('created', struct('max_peak_abs', obj.MaxPeak, 'max_output_power_dbm', obj.MaxOutputPower_dBm, ...
                'timeout_s', obj.Timeout, 'rf_off_after_measure', obj.RFOffAfterMeasure, 'mock', obj.IsMock));
        end

        function arm(obj, operator)
            %ARM A named person arms the session; until then nothing can be sent.
            %   arm(lab) uses the Operator given to the constructor; arm(lab, "Your Name") names the operator now.
            arguments
                obj (1,1) opendpd.lab.Session
                operator (1,1) string = ""
            end
            if obj.State == "tripped"
                error('opendpd:lab:SafetyViolation', 'The session is tripped (%s); make a new session.', obj.TripReason);
            end
            if obj.State ~= "disarmed"
                error('opendpd:lab:SafetyViolation', 'Cannot arm from state %s.', obj.State);
            end
            name = strtrim(operator);
            if strlength(name) == 0
                name = obj.Operator;
            end
            if strlength(name) == 0
                error('opendpd:lab:SafetyViolation', ...
                    'Arming needs the operator''s name (Operator=... or arm(lab, "name")); RF output stays off.');
            end
            if ~obj.IsMock && string(getenv(char(obj.Gate))) ~= "1"
                error('opendpd:lab:SafetyViolation', ['Real RF output is not permitted in this environment (%s is not ' ...
                    '"1"); only an approved laboratory session may set it. RF output stays off.'], obj.Gate);
            end
            obj.pulse('while arming');
            obj.Operator = name;
            obj.State = "armed";
            obj.log_('armed', struct('operator', char(name)));
        end

        function y = measure(obj, u, options)
            %MEASURE Check the limits, play u and capture. Returns the capture as a complex column vector.
            %   y = measure(lab, u, SampleRate=fs) calls MeasureFcn(u, fs); without SampleRate it calls MeasureFcn(u).
            %   RequestedPower_dBm states the output power for a ceiling check when there is no PowerCalibration; if both
            %   are given the larger one is checked.
            arguments
                obj (1,1) opendpd.lab.Session
                u {mustBeNumeric}
                options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
                options.RequestedPower_dBm (1,1) double {mustBeReal, mustBeFinite}
            end
            haveRate = isfield(options, 'SampleRate');
            if obj.State ~= "armed"
                suffix = '';
                if obj.State == "tripped"
                    suffix = sprintf(': %s', obj.TripReason);
                end
                error('opendpd:lab:SafetyViolation', 'measure requires an armed session (state %s%s).', obj.State, suffix);
            end
            [z, problem] = opendpd.lab.Session.toComplexColumn(u);
            if strlength(problem) > 0
                error('opendpd:lab:SafetyViolation', 'Refusing to play the signal: %s Nothing sent.', problem);
            end
            peak = max(abs(z));
            if peak > obj.MaxPeak
                error('opendpd:lab:SafetyViolation', 'Peak |signal| %.4g exceeds the limit %g; nothing sent.', peak, obj.MaxPeak);
            end
            power = obj.outputPower(z, options);
            if ~isnan(obj.MaxOutputPower_dBm)
                if isnan(power)
                    error('opendpd:lab:SafetyViolation', ['A power ceiling (%g dBm) is set, so the output power must be ' ...
                        'known: give the session a PowerCalibration or pass RequestedPower_dBm. Nothing sent.'], ...
                        obj.MaxOutputPower_dBm);
                end
                if power > obj.MaxOutputPower_dBm
                    error('opendpd:lab:SafetyViolation', 'Requested %g dBm exceeds the ceiling %g dBm; nothing sent.', ...
                        power, obj.MaxOutputPower_dBm);
                end
            end

            rate = NaN;
            if haveRate
                rate = options.SampleRate;
            end
            obj.PendingReason = "";
            obj.State = "active";
            obj.log_('measure', struct('n_samples', numel(z), 'peak', peak, 'sample_rate_hz', rate, 'power_dbm', power));
            release = onCleanup(@() obj.finishMeasure());       % the one place that trips: errors AND interrupts end here
            started = tic;
            try
                obj.pulse('before the measurement');
                if haveRate
                    captured = obj.MeasureFcn(z, rate);
                else
                    captured = obj.MeasureFcn(z);
                end
                elapsed = toc(started);
                if elapsed > obj.Timeout
                    error('opendpd:lab:Timeout', 'measure took %.3g s; the limit is %g s. RF output off.', elapsed, obj.Timeout);
                end
                obj.pulse('after the measurement');
                [y, problem] = opendpd.lab.Session.toComplexColumn(captured);
                if strlength(problem) > 0
                    error('opendpd:lab:BadCapture', 'The capture is unusable: %s', problem);
                end
            catch cause
                obj.PendingReason = string(cause.message);
                rethrow(cause);
            end
            obj.State = "armed";
            obj.Measurements(end+1, 1) = struct('index', numel(obj.Measurements) + 1, ...
                'at', char(opendpd.lab.Session.timestamp()), 'n_played', numel(z), 'n_captured', numel(y), ...
                'sample_rate_hz', rate, 'peak', peak, 'power_dbm', power, 'played_sha256', opendpd.lab.iqHash(z), ...
                'captured_sha256', opendpd.lab.iqHash(y), 'seconds', elapsed);
            obj.log_('measure_done', struct('seconds', elapsed));
            clear release
            if obj.RFOffAfterMeasure && ~obj.rfOffNow('end of the measurement')
                obj.State = "tripped";
                obj.TripReason = "RF off failed after a measurement";
                obj.log_('tripped', struct('reason', char(obj.TripReason)));
                error('opendpd:lab:RFOffFailed', ['RF off failed after the measurement; the output may still be on. ' ...
                    'Switch it off at the instrument now. The session is tripped.']);
            end
        end

        function disarm(obj)
            %DISARM RF off. A tripped session stays tripped.
            ok = obj.rfOffNow('disarm');
            if ok
                if obj.State ~= "tripped"
                    obj.State = "disarmed";
                end
                obj.log_('disarmed', struct());
            else
                obj.State = "tripped";
                if strlength(obj.TripReason) == 0
                    obj.TripReason = "RF off failed at disarm";
                end
                obj.log_('tripped', struct('reason', char(obj.TripReason)));
                error('opendpd:lab:RFOffFailed', ['RF off failed; the output may still be on. Switch it off at the ' ...
                    'instrument now. The session is tripped.']);
            end
        end

        function abort(obj, reason)
            %ABORT Manual stop: RF off and tripped, so nothing runs afterwards.
            arguments
                obj (1,1) opendpd.lab.Session
                reason (1,1) string = "abort requested"
            end
            obj.trip(reason);
        end

        function r = record(obj, options)
            %RECORD The session as a struct: limits, operator, the hash of every played and captured signal, the log of
            % every state change, and whether the instrument was a mock.
            %   record(lab) is the complete record. record(lab, Compact=true) is a short form that fits the 2000
            %   characters a dataset keeps as notes (free text is cut: the operator at 100 characters, the trip reason at
            %   200): use it as importIQ(..., Source=record(lab, Compact=true)), and keep the complete record with
            %   saveRecord (the compact form carries its SHA-256).
            arguments
                obj (1,1) opendpd.lab.Session
                options.Compact (1,1) logical = false
            end
            full = obj.fullRecord();
            if ~options.Compact
                r = full;
                return
            end
            reason = obj.TripReason;
            if strlength(reason) > 200
                reason = extractBefore(reason, 201) + "...";
            end
            operator = obj.Operator;
            if strlength(operator) > 100
                operator = extractBefore(operator, 101) + "...";
            end
            r = struct('format', char(obj.Format), 'compact', true, 'mock', obj.IsMock, 'operator', char(operator), ...
                'started_at', full.started_at, 'final_state', full.final_state, 'trip_reason', char(reason), ...
                'rf_off_failed', full.rf_off_failed, 'limits', full.limits, 'n_measurements', numel(obj.Measurements), ...
                'record_sha256', obj.recordHash());
            if ~isempty(obj.Measurements)
                last = obj.Measurements(end);
                r.last = struct('sample_rate_hz', last.sample_rate_hz, 'power_dbm', last.power_dbm, ...
                    'n_played', last.n_played, 'n_captured', last.n_captured, ...
                    'played_sha256', last.played_sha256, 'captured_sha256', last.captured_sha256);
            end
        end

        function info = saveRecord(obj, file)
            %SAVERECORD Write the complete record as JSON and return its file name and SHA-256.
            %   The SHA-256 is the record_sha256 of record(lab, Compact=true) if nothing happened in between.
            arguments
                obj (1,1) opendpd.lab.Session
                file (1,1) string
            end
            bytes = obj.recordBytes();
            folder = fileparts(file);
            if strlength(folder) > 0 && ~isfolder(folder)
                mkdir(folder);
            end
            fid = fopen(char(file), 'w');
            if fid < 0
                error('opendpd:lab:Record', 'Cannot write %s.', file);
            end
            closeFile = onCleanup(@() fclose(fid));
            fwrite(fid, bytes, 'uint8');
            info = struct('file', file, 'sha256', opendpd.internal.sha256Data(bytes), 'bytes', numel(bytes));
        end

        function delete(obj)
            if ~isempty(obj.RFOffFcn) && any(obj.State == ["armed", "active"])
                obj.rfOffNow('the session was deleted while armed');
            end
        end
    end

    methods (Access = protected)
        function header = getHeader(obj)
            if ~isscalar(obj)
                header = getHeader@matlab.mixin.CustomDisplay(obj);
                return
            end
            kind = 'RF-capable chain';
            if obj.IsMock
                kind = 'mock, emits nothing';
            end
            header = sprintf('  opendpd.lab.Session (%s) - %s, %d measurement(s)\n', kind, upper(char(obj.State)), numel(obj.Measurements));
        end

        function groups = getPropertyGroups(obj)
            if ~isscalar(obj)
                groups = getPropertyGroups@matlab.mixin.CustomDisplay(obj);
                return
            end
            names = {'State', 'Operator', 'TripReason', 'MaxPeak', 'MaxOutputPower_dBm', 'Timeout', 'RFOffAfterMeasure', 'RFOffFailed'};
            groups = matlab.mixin.util.PropertyGroup(names);
        end
    end

    methods (Access = private)
        function trip(obj, reason)
            obj.rfOffNow(reason);
            obj.State = "tripped";
            if strlength(obj.TripReason) == 0
                obj.TripReason = string(reason);
            end
            obj.log_('tripped', struct('reason', char(reason)));
        end

        function finishMeasure(obj)
            % Runs whenever measure is left. If the measurement did not finish (an error, or Ctrl+C), the session trips.
            if ~isvalid(obj) || obj.State ~= "active"
                return
            end
            reason = obj.PendingReason;
            if strlength(reason) == 0
                reason = "interrupted before the measurement finished";
            end
            obj.trip(reason);
        end

        function ok = rfOffNow(obj, reason)
            ok = true;
            try
                obj.RFOffFcn();
                obj.log_('rf_off', struct('reason', char(reason)));
            catch cause
                ok = false;
                obj.RFOffFailed = true;
                obj.log_('rf_off_failed', struct('reason', char(reason), 'error', cause.message));
                warning('opendpd:lab:RFOffFailed', ['RF off FAILED (%s). The output may still be on: switch it off at ' ...
                    'the instrument now.'], cause.message);
            end
        end

        function pulse(obj, when)
            if isempty(obj.HeartbeatFcn)
                return
            end
            try
                obj.HeartbeatFcn();
            catch cause
                error('opendpd:lab:LinkLost', 'Link to the instrument lost %s: %s', when, cause.message);
            end
        end

        function power = outputPower(obj, z, options)
            candidates = [];
            if ~isempty(obj.PowerCalibration)
                try
                    calibrated = double(obj.PowerCalibration(z));
                catch cause
                    error('opendpd:lab:SafetyViolation', 'The power calibration failed (%s); nothing sent.', cause.message);
                end
                if ~(isscalar(calibrated) && isreal(calibrated) && isfinite(calibrated))
                    error('opendpd:lab:SafetyViolation', ...
                        'The power calibration did not return one finite number of dBm; nothing sent.');
                end
                candidates(end+1) = calibrated; %#ok<AGROW>
            end
            if isfield(options, 'RequestedPower_dBm')
                candidates(end+1) = options.RequestedPower_dBm; %#ok<AGROW>
            end
            if isempty(candidates)
                power = NaN;
            else
                power = max(candidates);
            end
        end

        function log_(obj, event, detail)
            obj.Log(end+1, 1) = struct('At', char(opendpd.lab.Session.timestamp()), 'Event', event, ...
                'State', char(obj.State), 'Detail', detail);
        end

        function full = fullRecord(obj)
            ceiling = obj.MaxOutputPower_dBm;
            full = struct('format', char(obj.Format), ...
                'adapter', struct('name', char(obj.Name), 'description', char(obj.Description), 'mock', obj.IsMock), ...
                'mock', obj.IsMock, 'operator', char(obj.Operator), 'started_at', char(obj.StartedAt), ...
                'limits', struct('max_peak_abs', obj.MaxPeak, 'max_output_power_dbm', ceiling, 'timeout_s', obj.Timeout, ...
                    'rf_off_after_measure', obj.RFOffAfterMeasure), ...
                'final_state', char(obj.State), 'trip_reason', char(obj.TripReason), 'rf_off_failed', obj.RFOffFailed, ...
                'matlab_release', version('-release'), 'measurements', obj.Measurements, 'log', obj.Log);
        end

        function bytes = recordBytes(obj)
            bytes = unicode2native(jsonencode(obj.fullRecord(), PrettyPrint=true), 'UTF-8').';
        end

        function hash = recordHash(obj)
            hash = opendpd.internal.sha256Data(obj.recordBytes());
        end
    end

    methods (Static, Access = private)
        function text = timestamp()
            text = string(datetime('now', TimeZone='UTC', Format="yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"));
        end

        function [y, problem] = toComplexColumn(x)
            % A complex column from a complex or real vector, or from a real N-by-2 [I Q] matrix; PROBLEM says why not.
            y = [];
            problem = "";
            if ~isnumeric(x)
                problem = "it is not numeric.";
            elseif isempty(x)
                problem = "it is empty.";
            elseif isreal(x) && ismatrix(x) && size(x, 2) == 2 && size(x, 1) ~= 2
                y = double(complex(x(:, 1), x(:, 2)));
            elseif isvector(x)
                y = double(x(:));
            else
                problem = "it must be a vector or a real N-by-2 [I Q] matrix.";
            end
            if strlength(problem) == 0 && ~all(isfinite(y))
                y = [];
                problem = "it contains non-finite samples.";
            end
        end
    end
end
