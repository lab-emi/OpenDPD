classdef Model < matlab.System
    % OPENDPD.MODEL An opendpd-model-v1 package run by plain MATLAB code (no Python, no other toolbox).
    %   model = opendpd.load("pkg.opendpd.zip");
    %   y = opendpd.apply(model, x);                          % how the run was scored: state resets every nperseg samples
    %   y = opendpd.apply(model, x, Execution="streaming");   % one state across the waveform (gru, gmp only)
    %   y = model(chunk);  reset(model);                      % streaming, as a System object (same models)
    % Samples are not normalised or aligned: use the training dataset's units and sample rate (model.Manifest.signal).
    % Like the evaluator, input is rounded to single before it is used; apply returns a complex single column vector.
    % The kernels in opendpd.runtime are plain MATLAB. For MATLAB Coder and Simulink MATLAB System blocks use
    % opendpd.generateCode(model, folder), which writes a standalone class of the same kernels; HDL is not supported.
    properties (SetAccess = private)
        Manifest = struct()          % the package manifest (model, signal, scaling, execution, evidence, provenance)
        Source = ""                  % file the model was loaded from
        SHA256 = ""                  % SHA-256 of that file
    end
    properties (Access = private)
        GoldenData = struct()
        Kernel = struct('key', '')
        Hidden = []                  % recurrent state, H-by-L
        History = complex(zeros(0, 1))
    end
    properties (Constant, Hidden)
        MaxGoldenTolerance = 1e-5    % a package may ask for a tighter test, never a looser one
    end
    methods
        function obj = Model(package, source)
            if nargin > 0
                try
                    kernel = opendpd.Model.pack(package.manifest, package.weights);
                catch cause
                    if startsWith(cause.identifier, 'opendpd:')
                        rethrow(cause);
                    end
                    error('opendpd:Package', 'The package does not describe a model this toolbox supports: %s', cause.message);
                end
                obj.Manifest = package.manifest;
                obj.GoldenData = package.golden;
                obj.Source = string(source);
                obj.SHA256 = string(package.sha256);
                obj.Kernel = kernel;
            end
        end

        function [y, info] = apply(obj, x, options)
            %APPLY Run the model on a whole waveform with one of the two OpenDPD execution semantics.
            arguments
                obj (1,1) opendpd.Model
                x {mustBeNumeric}
                options.Execution (1,1) string {mustBeMember(options.Execution, ...
                    ["offline_segmented", "streaming_stateful", "streaming"])} = "offline_segmented"
                options.ChunkSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            obj.requireLoaded();
            try
                validateattributes(x, {'single', 'double'}, {'vector', 'nonempty', 'finite', 'nonsparse'});
            catch cause
                error('opendpd:InvalidIQ', 'Expected a finite single/double I/Q vector: %s', cause.message);
            end
            signal = complex(double(single(real(x(:)))), double(single(imag(x(:)))));
            execution = options.Execution;
            if execution == "streaming"
                execution = "streaming_stateful";
            end
            if execution == "offline_segmented"
                nperseg = obj.Manifest.signal.nperseg;
                count = numel(signal);
                segments = ceil(count / nperseg);
                padded = [signal; complex(zeros(segments * nperseg - count, 1))];
                out = complex(zeros(segments * nperseg, 1));
                for s = 1:segments
                    range = (s - 1) * nperseg + (1:nperseg);
                    out(range) = obj.segmentForward(padded(range));
                end
                yc = out(1:count);
            else
                obj.requireStreaming();
                chunk = options.ChunkSamples;
                if chunk == 0
                    chunk = numel(signal);
                end
                [hidden, history] = obj.initialState();
                yc = complex(zeros(numel(signal), 1));
                for first = 1:chunk:numel(signal)
                    range = first:min(first + chunk - 1, numel(signal));
                    [yc(range), hidden, history] = obj.streamForward(signal(range), hidden, history);
                end
            end
            y = complex(single(real(yc)), single(imag(yc)));
            if nargout > 1
                info = obj.describe(execution);
            end
        end

        function report = verify(obj)
            %VERIFY Run the package's golden test vector; see opendpd.verify.
            obj.requireLoaded();
            if ~isfield(obj.GoldenData, 'input') || ~isfield(obj.GoldenData, 'output_offline_segmented')
                error('opendpd:Package', 'This package has no golden test vector.');
            end
            g = obj.GoldenData;
            tolerance = min(double(obj.Manifest.golden.tolerance_abs), obj.MaxGoldenTolerance);
            x = complex(double(g.input(:, 1)), double(g.input(:, 2)));
            report = struct('model', obj.Kernel.key, 'samples', numel(x), 'tolerance_abs', tolerance, ...
                'offline_max_abs_error', NaN, 'streaming_max_abs_error', NaN, 'passed', false);
            report.offline_max_abs_error = opendpd.Model.maxError(obj.apply(x), g.output_offline_segmented);
            passed = report.offline_max_abs_error <= tolerance;
            if obj.Manifest.execution.streaming_stateful.available
                if ~isfield(g, 'output_streaming_stateful')
                    error('opendpd:Package', 'The manifest offers streaming but the package has no streaming golden output.');
                end
                chunk = obj.Manifest.golden.streaming_chunk_samples;
                s = obj.apply(x, Execution="streaming_stateful", ChunkSamples=chunk);
                report.streaming_max_abs_error = opendpd.Model.maxError(s, g.output_streaming_stateful);
                passed = passed && report.streaming_max_abs_error <= tolerance;
            end
            report.passed = passed;
        end

        function [coefficients, info] = commCoefficients(obj)
            %COMMCOEFFICIENTS The Coefficients matrix of comm.DPD (memory polynomial) for an mp_ls model.
            % coefficients = reshape(w, Q, K): row q+1 is lag q, column k+1 is envelope power k. Valid for mp_ls only.
            obj.requireLoaded();
            if ~strcmp(obj.Kernel.key, 'mp_ls')
                error('opendpd:NoMathWorksEquivalent', ['%s has no comm.DPD equivalent: only the memory polynomial ' ...
                    '(mp_ls) is the same polynomial; GMP terms and neural networks are not.'], obj.Kernel.key);
            end
            coefficients = reshape(obj.Kernel.coefficients, obj.Kernel.Q, obj.Kernel.K);
            info = struct('polynomial_type', 'Memory polynomial', 'degree', obj.Kernel.K, 'memory_depth', obj.Kernel.Q, ...
                'note', ['comm.DPD runs one continuous stream with a zero initial state; OpenDPD resets the state every ' ...
                'nperseg samples when it scores a run, so the two agree within one segment.']);
        end
    end

    methods (Hidden)
        function data = codegenInputs(obj)
            %CODEGENINPUTS What opendpd.generateCode writes a standalone class from: manifest, kernel arrays, golden vector.
            obj.requireLoaded();
            data = struct('Manifest', obj.Manifest, 'Kernel', obj.Kernel, 'Golden', obj.GoldenData, ...
                'SHA256', char(obj.SHA256));
        end
    end

    methods (Access = protected)
        function header = getHeader(obj)
            if isempty(obj.Kernel.key)
                header = sprintf('  opendpd.Model (empty)\n');
                return
            end
            m = obj.Manifest;
            header = sprintf(['  opendpd.Model  %s (%s), run %s\n    sample rate %g Hz, segment %d samples, evidence: %s, ' ...
                'streaming: %s\n'], obj.Kernel.key, m.run.role, m.run.run_id, m.signal.sample_rate_hz, m.signal.nperseg, ...
                m.evidence.type, string(m.execution.streaming_stateful.available));
        end

        function groups = getPropertyGroups(~)
            groups = matlab.mixin.util.PropertyGroup({'Manifest', 'Source', 'SHA256'});
        end

        function setupImpl(obj)
            obj.requireLoaded();
            obj.requireStreaming();
            [obj.Hidden, obj.History] = obj.initialState();
        end

        function resetImpl(obj)
            if ~isempty(obj.Kernel.key) && obj.Manifest.execution.streaming_stateful.available
                [obj.Hidden, obj.History] = obj.initialState();
            end
        end

        function y = stepImpl(obj, x)
            signal = complex(double(single(real(x(:)))), double(single(imag(x(:)))));
            [yc, obj.Hidden, obj.History] = obj.streamForward(signal, obj.Hidden, obj.History);
            y = complex(single(real(yc)), single(imag(yc)));
        end

        function validateInputsImpl(~, x)
            validateattributes(x, {'single', 'double'}, {'vector', 'nonempty', 'finite', 'nonsparse'});
        end

        function flag = isInputSizeMutableImpl(~, ~)
            flag = true;
        end

        function flag = isInputComplexityMutableImpl(~, ~)
            flag = true;
        end

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.OpenDPDModel = struct('Manifest', obj.Manifest, 'Source', obj.Source, 'SHA256', obj.SHA256, ...
                'GoldenData', obj.GoldenData, 'Kernel', obj.Kernel, 'Hidden', obj.Hidden, 'History', obj.History);
        end

        function loadObjectImpl(obj, s, wasLocked)
            d = s.OpenDPDModel;
            obj.Manifest = d.Manifest;
            obj.Source = d.Source;
            obj.SHA256 = d.SHA256;
            obj.GoldenData = d.GoldenData;
            obj.Kernel = d.Kernel;
            obj.Hidden = d.Hidden;
            obj.History = d.History;
            loadObjectImpl@matlab.System(obj, s, wasLocked);
        end
    end

    methods (Access = private)
        function requireLoaded(obj)
            if isempty(obj.Kernel.key)
                error('opendpd:Package', 'This opendpd.Model is empty; create one with opendpd.load.');
            end
        end

        function requireStreaming(obj)
            if ~obj.Manifest.execution.streaming_stateful.available
                error('opendpd:NoStreamingVariant', ['''%s'' has no registered streaming variant; use ' ...
                    'Execution="offline_segmented", which is how the run was scored.'], obj.Kernel.key);
            end
        end

        function [hidden, history] = initialState(obj)
            k = obj.Kernel;
            hidden = [];
            history = complex(zeros(0, 1));
            if strcmp(k.key, 'gru')
                hidden = zeros(size(k.weightHH{1}, 2), numel(k.weightIH));
            elseif strcmp(k.key, 'gmp')
                history = complex(zeros(2 * (k.M - 1), 1));
            end
        end

        function y = segmentForward(obj, segment)
            k = obj.Kernel;
            switch k.key
                case 'mp_ls'
                    y = opendpd.runtime.mpForward(segment, k.coefficients, k.K, k.Q);
                case 'gmp_ls'
                    y = opendpd.runtime.gmpPolynomialForward(segment, k.coefficients, k.params);
                case 'gmp'
                    y = opendpd.runtime.gmpForward(segment, k.weight, k.M, k.D);
                case 'gru'
                    y = opendpd.runtime.gruForward(segment, k, zeros(size(k.weightHH{1}, 2), numel(k.weightIH)));
                case 'tres_gru'
                    y = opendpd.runtime.tresGruForward(segment, k);
                otherwise
                    error('opendpd:Package', 'Model "%s" is not supported by this toolbox version.', k.key);
            end
        end

        function [y, hidden, history] = streamForward(obj, chunk, hidden, history)
            k = obj.Kernel;
            switch k.key
                case 'gru'
                    [y, hidden] = opendpd.runtime.gruForward(chunk, k, hidden);
                case 'gmp'
                    block = [history; chunk];
                    full = opendpd.runtime.gmpForward(block, k.weight, k.M, k.D);
                    y = full(numel(history) + 1:end);
                    keep = numel(history);
                    history = block(end - keep + 1:end);
                otherwise
                    obj.requireStreaming();
                    y = chunk;
            end
        end

        function info = describe(obj, execution)
            m = obj.Manifest;
            info = struct('execution', char(execution), 'model', m.model.key, 'run_id', m.run.run_id, ...
                'output_role', 'modeled_pa_output', 'sample_rate_hz', m.signal.sample_rate_hz, ...
                'segment_samples', m.signal.nperseg, 'lookahead_samples', m.execution.offline_segmented.lookahead_samples, ...
                'evidence', m.evidence.type, 'runtime', 'plain MATLAB (opendpd.runtime)', ...
                'note', 'no normalisation or alignment is applied; use the training dataset''s units and sample rate');
            if strcmp(m.run.role, 'dpd')
                info.output_role = 'predistorted_pa_input';
            end
            if execution == "streaming_stateful"
                info.segment_samples = [];
                info.lookahead_samples = 0;
            end
        end
    end

    methods (Static, Access = private)
        function kernel = pack(manifest, w)
            % Turn the manifest and the weight arrays into the kernel's inputs, refusing anything that is not exactly
            % the shape the model needs (a hostile or damaged package must fail here, not inside a kernel).
            opendpd.Model.requireFields(manifest, ["model.key", "model.parameters", "model.architecture", "run.run_id", ...
                "run.role", "signal.sample_rate_hz", "signal.nperseg", "execution.offline_segmented.lookahead_samples", ...
                "execution.streaming_stateful.available", "evidence.type", "golden.tolerance_abs"]);
            key = manifest.model.key;
            kernel = struct('key', key);
            integer = @(name, v, lo) opendpd.Model.mustBeCount(name, v, lo);
            nperseg = integer('signal.nperseg', manifest.signal.nperseg, 1);
            if nperseg > 1e8
                error('opendpd:Package', 'signal.nperseg = %g is not a plausible segment length.', nperseg);
            end
            streaming = manifest.execution.streaming_stateful;
            if ~(islogical(streaming.available) && isscalar(streaming.available))
                error('opendpd:Package', 'execution.streaming_stateful.available must be true or false.');
            end
            switch key
                case 'mp_ls'
                    K = integer('K', manifest.model.parameters.K, 1);
                    Q = integer('Q', manifest.model.parameters.Q, 1);
                    kernel.K = K;
                    kernel.Q = Q;
                    kernel.coefficients = complex(opendpd.Model.vector(w, 'coefficients', K * Q));
                    if streaming.available
                        error('opendpd:Package', 'mp_ls has no streaming variant; the manifest says otherwise.');
                    end
                case 'gmp_ls'
                    p = manifest.model.parameters;
                    names = ["Ka", "La", "Kb", "Lb", "Mb", "Kc", "Lc", "Mc"];
                    kernel.params = struct();
                    for name = names
                        kernel.params.(name) = integer(char(name), p.(name), 0);
                    end
                    q = kernel.params;
                    count = q.Ka * q.La + q.Kb * q.Lb * q.Mb + q.Kc * q.Lc * q.Mc;
                    kernel.coefficients = complex(opendpd.Model.vector(w, 'coefficients', count));
                    if streaming.available
                        error('opendpd:Package', 'gmp_ls has no streaming variant; the manifest says otherwise.');
                    end
                case 'gmp'
                    kernel.M = integer('memory_length', manifest.model.architecture.memory_length, 1);
                    kernel.D = integer('degree', manifest.model.architecture.degree, 1);
                    kernel.weight = double(opendpd.Model.vector(w, 'gmp_weight', kernel.M * (1 + (kernel.D - 1) * kernel.M)));
                    if streaming.available
                        if ~(isfield(streaming, 'history_samples') && isequal(streaming.history_samples, 2 * (kernel.M - 1)))
                            error('opendpd:Package', ['The gmp streaming history must be 2*(memory_length-1) = %d samples ' ...
                                'and the manifest says otherwise.'], 2 * (kernel.M - 1));
                        end
                    end
                case {'gru', 'tres_gru'}
                    hidden = integer('hidden_size', manifest.model.architecture.hidden_size, 1);
                    layers = integer('num_layers', manifest.model.architecture.num_layers, 1);
                    features = 2;
                    if strcmp(key, 'tres_gru')
                        features = 6;
                        if streaming.available
                            error('opendpd:Package', 'tres_gru has no streaming variant; the manifest says otherwise.');
                        end
                    end
                    kernel.weightIH = cell(1, layers);
                    kernel.weightHH = cell(1, layers);
                    kernel.biasIH = cell(1, layers);
                    kernel.biasHH = cell(1, layers);
                    for l = 1:layers
                        in = features;
                        if l > 1
                            in = hidden;
                        end
                        kernel.weightIH{l} = opendpd.Model.matrix(w, sprintf('rnn_weight_ih_l%d', l - 1), [3 * hidden, in]);
                        kernel.weightHH{l} = opendpd.Model.matrix(w, sprintf('rnn_weight_hh_l%d', l - 1), [3 * hidden, hidden]);
                        kernel.biasIH{l} = opendpd.Model.optionalVector(w, sprintf('rnn_bias_ih_l%d', l - 1), 3 * hidden);
                        kernel.biasHH{l} = opendpd.Model.optionalVector(w, sprintf('rnn_bias_hh_l%d', l - 1), 3 * hidden);
                    end
                    kernel.fcWeight = opendpd.Model.matrix(w, 'fc_weight', [2, hidden]);
                    kernel.fcBias = opendpd.Model.optionalVector(w, 'fc_bias', 2);
                    if strcmp(key, 'tres_gru')
                        kernel.conv1 = opendpd.Model.matrix(w, 'tcn_conv1_weight', [3, 2, 3]);
                        kernel.conv2 = opendpd.Model.matrix(w, 'tcn_conv2_weight', [2, 3, 1]);
                    end
                otherwise
                    error('opendpd:Package', 'Model "%s" is not supported by this toolbox version.', key);
            end
            if streaming.available
                chunk = integer('golden.streaming_chunk_samples', manifest.golden.streaming_chunk_samples, 1); %#ok<NASGU>
            end
        end

        function requireFields(manifest, paths)
            for path = paths
                node = manifest;
                for part = split(path, ".").'
                    if ~isstruct(node) || ~isfield(node, part)
                        error('opendpd:Package', 'The manifest has no "%s".', path);
                    end
                    node = node.(part);
                end
            end
            if ~ischar(manifest.model.key) || ~ischar(manifest.run.role) || ~ischar(manifest.run.run_id) || ...
                    ~ischar(manifest.evidence.type)
                error('opendpd:Package', 'model.key, run.role, run.run_id and evidence.type must be text.');
            end
            if ~(isnumeric(manifest.signal.sample_rate_hz) && isscalar(manifest.signal.sample_rate_hz) ...
                    && isfinite(manifest.signal.sample_rate_hz) && manifest.signal.sample_rate_hz > 0)
                error('opendpd:Package', 'signal.sample_rate_hz must be a positive number.');
            end
            if ~(isnumeric(manifest.golden.tolerance_abs) && isscalar(manifest.golden.tolerance_abs) ...
                    && manifest.golden.tolerance_abs > 0)
                error('opendpd:Package', 'golden.tolerance_abs must be a positive number.');
            end
        end

        function value = mustBeCount(name, v, lower)
            if ~(isnumeric(v) && isscalar(v) && isfinite(v) && v == fix(v) && v >= lower && v <= 2^31)
                error('opendpd:Package', '%s must be an integer of at least %d (got %s).', name, lower, mat2str(v));
            end
            value = double(v);
        end

        function value = vector(w, name, count)
            if ~isfield(w, name) || ~isnumeric(w.(name)) || numel(w.(name)) ~= count || ~(isvector(w.(name)) || count <= 1)
                error('opendpd:Package', 'Weight "%s" must be a vector of %d numbers.', name, count);
            end
            value = double(w.(name)(:));
            if ~isreal(value) && ~strcmp(name, 'coefficients')
                error('opendpd:Package', 'Weight "%s" must be real.', name);
            end
        end

        function value = matrix(w, name, expected)
            if ~isfield(w, name) || ~isnumeric(w.(name))
                error('opendpd:Package', 'The package has no numeric weight "%s".', name);
            end
            actual = size(w.(name));
            expected = expected(:).';
            actual(end+1:numel(expected)) = 1;
            expected(end+1:numel(actual)) = 1;
            if ~isequal(actual, expected) || ~isreal(w.(name))
                error('opendpd:Package', 'Weight "%s" has size %s; this model needs real %s.', name, mat2str(size(w.(name))), mat2str(expected));
            end
            value = double(w.(name));
        end

        function value = optionalVector(w, name, count)
            if isfield(w, name)
                value = opendpd.Model.vector(w, name, count);
            else
                value = zeros(count, 1);
            end
        end

        function e = maxError(y, reference)
            % NaN-safe: max() ignores NaN, so a model that returns NaN must not look accurate.
            reference = double(reference);
            d = [real(double(y(:))) - reference(:, 1); imag(double(y(:))) - reference(:, 2)];
            if numel(y) ~= size(reference, 1) || any(~isfinite(d))
                e = Inf;
            else
                e = max(abs(d));
            end
        end
    end
end
