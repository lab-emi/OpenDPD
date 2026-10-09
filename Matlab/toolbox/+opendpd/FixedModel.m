classdef FixedModel < matlab.mixin.CustomDisplay
    % OPENDPD.FIXEDMODEL A fixed-point-v1 deployment package run bit for bit by plain MATLAB code (no Python, no other toolbox).
    %   model = opendpd.load("run-deploy.zip");   % a package written by `opendpd deploy` or the Studio's Deployment panel
    %   report = opendpd.verify(model);           % replays the six golden vectors; every output sample and every state step
    %   y = opendpd.apply(model, x);              % float I/Q in, float I/Q out, through the integer model
    %   [yq, state, trace] = runInteger(model, xq);   % integers in, integers out, the state after every sample
    % The model is the one-layer GRU of docs/protocols/fixed-point-v1.md, executed as the streaming reference does: one sample
    % per step, the state carried, every stored quantity saturated, every right shift a round-half-up floor. The integers are
    % held in double precision, which is exact because the package is checked at load so that no value can reach 2^53. This
    % is a numerical reference for checking an implementation, not a deployment: it is not HDL-ready and was not run on
    % hardware. Every format is read from the package, never assumed.
    properties (SetAccess = private)
        Manifest = struct()          % the package manifest (specification, run, golden index, verification, report)
        Source = ""                  % file the model was loaded from
        SHA256 = ""                  % SHA-256 of that file
    end
    properties (Access = private)
        Format = struct('hidden', 0)  % formats and integer weights as opendpd.runtime.fixedGruRun takes them
        Cases = struct([])            % the golden vectors: id, resets, x, y, h (state after every sample)
    end
    properties (Constant, Hidden)
        % The rule texts this toolbox implements; a package that says something else is refused (a changed rule is a new
        % specification id). tests/unit/test_fixed_point_package.py checks them against opendpd/schemas/fixed_point.py.
        Rounding = 'round half up: add 2^(s-1) then arithmetic shift right by s; left shifts are exact'
        Saturation = ['every stored quantity saturates to its word width (x, h, y, gate values, table index); ' ...
            'the accumulator never wraps']
        Nonlinearity = ['table lookup without interpolation: index = saturate(round(pre-activation to LUT_FRAC)) + ' ...
            'offset']
    end
    methods
        function obj = FixedModel(package, source)
            if nargin > 0
                try
                    [format, cases] = packModel(package);
                catch cause
                    if startsWith(cause.identifier, 'opendpd:')
                        rethrow(cause);
                    end
                    error('opendpd:Package', 'The package does not describe a model this toolbox supports: %s', cause.message);
                end
                obj.Manifest = package.manifest;
                obj.Source = string(source);
                obj.SHA256 = string(package.sha256);
                obj.Format = format;
                obj.Cases = cases;
            end
        end

        function [yq, state, trace] = runInteger(obj, xq, options)
            %RUNINTEGER Integers in, integers out: the reference executed on quantised samples.
            %   yq = runInteger(model, xq) runs the rows of XQ (N-by-2 integers in the input format; larger ones saturate, as in the
            %   reference) from the zero state. [yq, state, trace] also returns the final state and the state after every
            %   sample. Name-value pairs: State (the state to start from, default zeros) and ResetAt (1-based sample
            %   numbers before which the state is zeroed). Results are integer classes wide enough for the formats.
            arguments
                obj (1,1) opendpd.FixedModel
                xq {mustBeNumeric, mustBeReal, mustBeFinite, mustBeInteger}
                options.State {mustBeNumeric, mustBeReal, mustBeFinite, mustBeInteger} = []
                options.ResetAt (1,:) double {mustBeInteger, mustBePositive} = zeros(1, 0)
            end
            obj.requireLoaded();
            f = obj.Format;
            if ~ismatrix(xq) || size(xq, 2) ~= 2 || size(xq, 1) < 1
                error('opendpd:InvalidIQ', 'Expected an N-by-2 matrix of integers (I and Q), N >= 1.');
            end
            state = zeros(f.hidden, 1);
            if ~isempty(options.State)
                state = double(options.State(:));
                if numel(state) ~= f.hidden || any(state < f.hMin | state > f.hMax)
                    error('opendpd:InvalidState', 'State must be %d integers in the range of the state format [%d, %d].', ...
                        f.hidden, f.hMin, f.hMax);
                end
            end
            [y, state, trace] = opendpd.runtime.fixedGruRun(double(xq), state, options.ResetAt, f);
            yq = cast(y, f.yClass);
            state = cast(state, f.hClass);
            trace = cast(trace, f.hClass);
        end

        function [y, info] = apply(obj, x, options)
            %APPLY Run a waveform through the integer model; returns a complex single column vector.
            %   Like opendpd.apply for the float models, the input is rounded to single and is not normalised. It is then
            %   quantised to the input format (round half away from zero, saturating at the format's range) and the output
            %   integers are scaled back by the output format. The state starts at zero; Execution is the streaming reference,
            %   the only one this model has. ChunkSamples processes the waveform in chunks with the state carried; the result
            %   does not depend on it.
            arguments
                obj (1,1) opendpd.FixedModel
                x {mustBeNumeric}
                options.Execution (1,1) string {mustBeMember(options.Execution, ...
                    ["", "streaming_stateful", "streaming", "offline_segmented"])} = ""
                options.ChunkSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            obj.requireLoaded();
            if options.Execution == "offline_segmented"
                error('opendpd:NoOfflineVariant', ['A fixed-point model has one execution, the streaming reference of ' ...
                    'fixed-point-v1 (state carried across samples); there is no offline_segmented variant.']);
            end
            try
                validateattributes(x, {'single', 'double'}, {'vector', 'nonempty', 'finite', 'nonsparse'});
            catch cause
                error('opendpd:InvalidIQ', 'Expected a finite single/double I/Q vector: %s', cause.message);
            end
            f = obj.Format;
            q = obj.quantiseInput(x);
            count = size(q, 1);
            chunk = options.ChunkSamples;
            if chunk == 0
                chunk = count;
            end
            out = zeros(count, 2);
            state = zeros(f.hidden, 1);
            for first = 1:chunk:count
                range = first:min(first + chunk - 1, count);
                [out(range, :), state] = opendpd.runtime.fixedGruRun(q(range, :), state, zeros(1, 0), f);
            end
            out = out / 2^f.yFrac;
            y = complex(single(out(:, 1)), single(out(:, 2)));
            if nargout > 1
                info = struct('execution', 'streaming_stateful', 'model', 'gru', 'spec_id', obj.Manifest.spec.spec_id, ...
                    'run_id', obj.Manifest.run_id, ...
                    'input_format', sprintf('%d-bit, %d fractional bits (full scale +-%g)', f.xBits, f.xFrac, f.xMax / 2^f.xFrac), ...
                    'output_format', sprintf('%d-bit, %d fractional bits (full scale +-%g)', f.yBits, f.yFrac, f.yMax / 2^f.yFrac), ...
                    'runtime', 'plain MATLAB (opendpd.runtime.fixedGruRun), integers exact in double precision', ...
                    'note', ['no normalisation or alignment is applied; samples beyond the input format saturate; this is a ' ...
                    'numerical reference, not a deployment']);
            end
        end

        function report = verify(obj)
            %VERIFY Replay every golden vector and compare every output sample and every state step; see opendpd.verify.
            obj.requireLoaded();
            f = obj.Format;
            report = struct('model', 'gru', 'spec_id', obj.Manifest.spec.spec_id, 'status', 'bit_exact', 'passed', true, ...
                'cases_checked', 0, 'samples_checked', 0, 'mismatch_case', '', 'mismatch_sample', [], 'mismatch_signal', '', ...
                'package_c99_status', obj.Manifest.verification.status);
            for k = 1:numel(obj.Cases)
                c = obj.Cases(k);
                [y, ~, trace] = opendpd.runtime.fixedGruRun(double(c.x), zeros(f.hidden, 1), c.resets, f);
                badState = find(any(trace ~= double(c.h), 2), 1);
                badOutput = find(any(y ~= double(c.y), 2), 1);
                if ~isempty(badState) || ~isempty(badOutput)
                    % The state is compared before the output of the same step, because the output is computed from it.
                    if ~isempty(badState) && (isempty(badOutput) || badState <= badOutput)
                        report.mismatch_sample = badState;
                        report.mismatch_signal = 'h';
                    else
                        report.mismatch_sample = badOutput;
                        report.mismatch_signal = 'y';
                    end
                    report.status = 'mismatch';
                    report.passed = false;
                    report.mismatch_case = c.id;
                    return
                end
                report.cases_checked = report.cases_checked + 1;
                report.samples_checked = report.samples_checked + size(c.x, 1);
            end
        end
    end

    methods (Hidden)
        function f = kernelFormat(obj)
            %KERNELFORMAT The formats and integer weights that opendpd.runtime.fixedGruRun takes (for tests and tools).
            obj.requireLoaded();
            f = obj.Format;
        end

        function q = quantiseInput(obj, x)
            %QUANTISEINPUT The integers apply feeds the model: I/Q rounded to single, times 2^frac, rounded half away from zero
            % and saturated to the input format, as an N-by-2 double matrix.
            obj.requireLoaded();
            f = obj.Format;
            scaled = [double(single(real(x(:)))), double(single(imag(x(:))))] * 2^f.xFrac;
            q = zeros(size(scaled));
            positive = scaled >= 0;
            q(positive) = floor(scaled(positive) + 0.5);
            q(~positive) = ceil(scaled(~positive) - 0.5);
            q = min(max(q, f.xMin), f.xMax);
        end
    end

    methods (Access = protected)
        function header = getHeader(obj)
            if obj.Format.hidden == 0
                header = sprintf('  opendpd.FixedModel (empty)\n');
                return
            end
            f = obj.Format;
            header = sprintf(['  opendpd.FixedModel  %s, run %s\n    GRU, %d hidden units; input %d-bit/%d, state %d-bit/%d, ' ...
                'output %d-bit/%d; package C99 verdict: %s\n'], obj.Manifest.spec.spec_id, obj.Manifest.run_id, f.hidden, ...
                f.xBits, f.xFrac, f.hBits, f.hFrac, f.yBits, f.yFrac, obj.Manifest.verification.status);
        end

        function groups = getPropertyGroups(~)
            groups = matlab.mixin.util.PropertyGroup({'Manifest', 'Source', 'SHA256'});
        end
    end

    methods (Access = private)
        function requireLoaded(obj)
            if obj.Format.hidden == 0
                error('opendpd:Package', 'This opendpd.FixedModel is empty; create one with opendpd.load.');
            end
        end
    end
end

% ----- turning a read package into the kernel's inputs ---------------------------------------------------------------
% A hostile or damaged package must fail here, with a message, not inside a kernel; and every value the kernel can form must
% be an integer below 2^53, which is what makes double precision exact.

function [f, cases] = packModel(package)
manifest = package.manifest;
spec = package.spec;
w = package.weights;
requireFields(manifest, ["run_id", "hidden_size", "verification"], 'the manifest');
if ~(ischar(manifest.run_id) && ~isempty(regexp(manifest.run_id, '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$', 'once')))
    error('opendpd:Package', 'The manifest''s run_id is not a plain identifier.');
end
requireFields(manifest.verification, "status", 'the manifest''s verification');
if ~(ischar(manifest.verification.status) && any(strcmp(manifest.verification.status, {'bit_exact', 'mismatch', 'not_run'})))
    error('opendpd:Package', 'The manifest''s verification status is not one of bit_exact, mismatch, not_run.');
end
requireFields(spec, ["spec_id", "model_key", "x", "h", "y", "pre", "weight_bits", "accumulator_bits", "sigmoid", "tanh", ...
    "rounding", "saturation", "nonlinearity"], 'the specification');
if ~strcmp(spec.model_key, 'gru_stream')
    error('opendpd:Package', 'The specification executes "%s"; this toolbox implements gru_stream.', shown(spec.model_key));
end
if ~isequal(spec.rounding, opendpd.FixedModel.Rounding) || ~isequal(spec.saturation, opendpd.FixedModel.Saturation) ...
        || ~isequal(spec.nonlinearity, opendpd.FixedModel.Nonlinearity)
    error('opendpd:Package', ['The package states rounding, saturation or table rules that differ from the ones this ' ...
        'toolbox implements (docs/protocols/fixed-point-v1.md); a changed rule is a new specification id.']);
end
requireFields(w, ["spec_id", "hidden", "inputs", "outputs", "fractions", "w_ih", "w_hh", "w_out", "b_ih", "b_hh", "b_out", ...
    "sigmoid_table", "tanh_table", "gate_order"], 'weights.json');
x = wordFormat('x', spec.x, 16);
h = wordFormat('h', spec.h, 16);
y = wordFormat('y', spec.y, 16);
pre = wordFormat('pre', spec.pre, 32);
weightBits = wholeNumber('weight_bits', spec.weight_bits, 4, 16);
accBits = wholeNumber('accumulator_bits', spec.accumulator_bits, 32, 53);
sigmoid = tableFormat('sigmoid', spec.sigmoid, w, 'sigmoid_table');
tanhTable = tableFormat('tanh', spec.tanh, w, 'tanh_table');
if ~strcmp(w.spec_id, 'fixed-point-v1')
    error('opendpd:Package', 'weights.json is not for fixed-point-v1.');
end
hidden = wholeNumber('hidden', w.hidden, 1, 4096);
if ~isequal(manifest.hidden_size, hidden)
    error('opendpd:Package', 'The manifest says %s hidden units and weights.json says %d.', shown(manifest.hidden_size), hidden);
end
if ~isequal(w.inputs, 2) || ~isequal(w.outputs, 2)
    error('opendpd:Package', 'A deployment package takes I/Q samples: inputs and outputs must both be 2.');
end
if ~isequal(w.gate_order, {'r'; 'z'; 'n'})
    error('opendpd:Package', 'weights.json lists the gates in an order other than r, z, n.');
end
requireFields(w.fractions, ["w_ih", "w_hh", "w_out", "bias"], 'fractions');
fractions = [wholeNumber('fractions.w_ih', w.fractions.w_ih, 0, 62), wholeNumber('fractions.w_hh', w.fractions.w_hh, 0, 62), ...
    wholeNumber('fractions.w_out', w.fractions.w_out, 0, 62)];
if ~isequal(w.fractions.bias, pre.frac)
    error('opendpd:Package', 'The bias fraction must equal the pre-activation fraction (%d).', pre.frac);
end
limit = 2^(weightBits - 1);
wIH = integerMatrix('w_ih', w.w_ih, [3 * hidden, 2], -limit, limit - 1);
wHH = integerMatrix('w_hh', w.w_hh, [3 * hidden, hidden], -limit, limit - 1);
wOut = integerMatrix('w_out', w.w_out, [2, hidden], -limit, limit - 1);
bIH = integerMatrix('b_ih', w.b_ih, [3 * hidden, 1], pre.min, pre.max);
bHH = integerMatrix('b_hh', w.b_hh, [3 * hidden, 1], pre.min, pre.max);
bOut = integerMatrix('b_out', w.b_out, [2, 1], pre.min, pre.max);
checkTensors(manifest, fractions, pre.frac);

f = struct('hidden', hidden, 'outputs', 2, 'accBits', accBits, 'accLimit', 2^(accBits - 1), ...
    'xBits', x.bits, 'xFrac', x.frac, 'xMin', x.min, 'xMax', x.max, ...
    'hBits', h.bits, 'hFrac', h.frac, 'hMin', h.min, 'hMax', h.max, 'hClass', wordClass(h.bits), ...
    'yBits', y.bits, 'yFrac', y.frac, 'yMin', y.min, 'yMax', y.max, 'yClass', wordClass(y.bits), ...
    'preBits', pre.bits, 'preFrac', pre.frac, 'preMin', pre.min, 'preMax', pre.max, ...
    'fIH', fractions(1), 'fHH', fractions(2), 'fOut', fractions(3), ...
    'wIH', wIH, 'wHH', wHH, 'wOut', wOut, 'bIH', bIH, 'bHH', bHH, 'bOut', bOut, 'sigmoid', sigmoid, 'tanh', tanhTable);
requireExact(f);
cases = goldenCases(manifest, package.golden, f);
end

function requireFields(s, names, label)
if ~isstruct(s)
    error('opendpd:Package', '%s must be an object.', label);
end
for name = names
    if ~isfield(s, name)
        error('opendpd:Package', '%s lacks "%s".', label, name);
    end
end
end

function text = shown(value)
% A value from the package, made safe to put in an error message.
if ischar(value) || isstring(value)
    text = regexprep(char(string(value)), '[^A-Za-z0-9_. -]', '?');
    text = text(1:min(numel(text), 60));
elseif isnumeric(value) && isscalar(value) && isreal(value)
    text = sprintf('%g', value);
else
    text = '?';
end
end

function value = wholeNumber(name, value, low, high)
if ~(isnumeric(value) && isscalar(value) && isreal(value) && isfinite(value) && value == fix(value) ...
        && value >= low && value <= high)
    error('opendpd:Package', '%s must be a whole number from %d to %d.', name, low, high);
end
value = double(value);
end

function word = wordFormat(name, s, maxBits)
requireFields(s, ["bits", "frac"], name);
bits = wholeNumber([name '.bits'], s.bits, 2, maxBits);
frac = wholeNumber([name '.frac'], s.frac, 0, 62);
word = struct('bits', bits, 'frac', frac, 'min', -2^(bits - 1), 'max', 2^(bits - 1) - 1);
end

function table = tableFormat(function_, s, weights, field)
requireFields(s, ["function", "range", "index_frac", "value"], function_);
if ~strcmp(s.function, function_)
    error('opendpd:Package', 'The %s table is declared as "%s".', function_, shown(s.function));
end
indexFrac = wholeNumber([function_ '.index_frac'], s.index_frac, 0, 16);
value = wordFormat([function_ '.value'], s.value, 16);
offset = s.range * 2^indexFrac;
if ~(isnumeric(offset) && isscalar(offset) && isfinite(offset) && offset == fix(offset) && offset >= 1 && offset <= 2^20)
    error('opendpd:Package', 'The %s table range times 2^index_frac must be a whole number of entries.', function_);
end
values = integerMatrix(field, weights.(field), [2 * offset, 1], value.min, value.max);
table = struct('values', values, 'range', offset, 'indexFrac', indexFrac);
end

function a = integerMatrix(name, a, expected, low, high)
if ~(isnumeric(a) && isreal(a) && all(isfinite(a(:))) && all(a(:) == fix(a(:))))
    error('opendpd:Package', '%s must hold whole numbers.', name);
end
if ~isequal(size(a), expected)
    error('opendpd:Package', '%s has size %s; %s is required.', name, mat2str(size(a)), mat2str(expected));
end
if any(a(:) < low) || any(a(:) > high)
    error('opendpd:Package', '%s has values outside [%d, %d].', name, low, high);
end
a = double(a);
end

function checkTensors(manifest, weightFractions, biasFraction)
% The manifest records how each tensor was quantised; it must agree with weights.json.
if ~isfield(manifest, 'tensors') || ~isstruct(manifest.tensors)
    error('opendpd:Package', 'The manifest does not describe the quantised tensors.');
end
expected = {'w_ih', weightFractions(1); 'w_hh', weightFractions(2); 'w_out', weightFractions(3); 'b_ih', biasFraction; ...
    'b_hh', biasFraction; 'b_out', biasFraction};
for k = 1:size(expected, 1)
    at = find(strcmp({manifest.tensors.name}, expected{k, 1}), 1);
    if isempty(at) || ~isequal(manifest.tensors(at).frac, expected{k, 2})
        error('opendpd:Package', 'The manifest and weights.json disagree about the fraction of %s.', expected{k, 1});
    end
end
end

function name = wordClass(bits)
if bits <= 8
    name = 'int8';
elseif bits <= 16
    name = 'int16';
else
    name = 'int32';
end
end

function requireExact(f)
% Every value the kernel forms is an integer, and double precision holds integers exactly below 2^53. These bounds follow from
% the formats and the actual weights, so no input the kernel can be given gets past them.
rowSum = @(w) max(sum(abs(w), 2));
sigmoidMax = max(abs(f.sigmoid.values));
tanhMax = max(abs(f.tanh.values));
shift = max([f.preFrac - (f.fIH + f.xFrac), f.preFrac - (f.fHH + f.hFrac), f.preFrac - (f.fOut + f.hFrac), 0]);
stages = {'the input accumulator', rowSum(f.wIH) * 2^(f.xBits - 1); ...
    'the recurrent accumulator', rowSum(f.wHH) * 2^(f.hBits - 1); ...
    'the output accumulator', rowSum(f.wOut) * 2^(f.hBits - 1); ...
    'an accumulator after its left shift', f.accLimit * 2^shift + 2^f.preBits; ...
    'the reset-gate product', sigmoidMax * 2^(f.preBits - 1); ...
    'the state update', (2^f.hFrac + sigmoidMax) * tanhMax + sigmoidMax * 2^(f.hBits - 1)};
for k = 1:size(stages, 1)
    if stages{k, 2} >= 2^53
        error('opendpd:Package', ['%s could reach %g, which double precision cannot hold exactly (the limit is 2^53); the ' ...
            'toolbox refuses the package instead of computing something that is not bit-exact.'], stages{k, 1}, stages{k, 2});
    end
end
end

function cases = goldenCases(manifest, golden, f)
index = manifest.golden;
if ~isstruct(index) || isempty(index) || ~isfield(index, 'case_id') || ~isfield(index, 'n_samples') ...
        || ~isfield(index, 'resets_at')
    error('opendpd:Package', 'The manifest does not index the golden vectors.');
end
ids = string({index.case_id});
if numel(unique(ids)) ~= numel(ids) || ~isequal(sort(ids(:)), sort(string(fieldnames(golden))))
    error('opendpd:Package', 'The golden vectors in the archive and the manifest''s index are not the same set.');
end
cases = struct('id', {}, 'resets', {}, 'x', {}, 'y', {}, 'h', {});
for k = 1:numel(index)
    id = char(ids(k));
    g = golden.(id);
    needed = ["x", "y", "h_trace", "h_final"];
    if ~all(isfield(g, needed))
        error('opendpd:Package', 'Golden case %s lacks one of x, y, h_trace, h_final.', shown(id));
    end
    n = wholeNumber('n_samples', index(k).n_samples, 1, 2^31);
    resets = reshape(double(index(k).resets_at), 1, []);
    if ~isempty(resets) && (any(resets ~= fix(resets)) || any(resets < 0) || any(resets >= n))
        error('opendpd:Package', 'Golden case %s resets the state at samples outside the vector.', shown(id));
    end
    if numel(g.x) ~= 2 * n || numel(g.y) ~= 2 * n || numel(g.h_trace) ~= f.hidden * n || numel(g.h_final) ~= f.hidden
        error('opendpd:Package', 'Golden case %s does not have the sizes its index declares.', shown(id));
    end
    trace = reshape(g.h_trace, f.hidden, n).';
    if ~isequal(trace(end, :).', g.h_final(:))
        error('opendpd:Package', 'Golden case %s: the final state differs from the last step of the trace.', shown(id));
    end
    cases(k).id = id;
    cases(k).resets = resets + 1;
    cases(k).x = reshape(g.x, 2, n).';
    cases(k).y = reshape(g.y, 2, n).';
    cases(k).h = trace;
end
end
