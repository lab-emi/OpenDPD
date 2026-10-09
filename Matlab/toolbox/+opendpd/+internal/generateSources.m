function sources = generateSources(data, name, execution)
%GENERATESOURCES The text of the files opendpd.generateCode writes for a loaded model package.
% DATA is opendpd.Model.codegenInputs, NAME the class name (already checked), EXECUTION "offline_segmented" or
% "streaming_stateful". Returns a struct of char row vectors: class, step (the MATLAB Coder entry point), check (the golden
% test) and readme. A pure function of its inputs: the same package gives the same bytes. Nothing from the manifest reaches
% the code except numbers formatted here and text reduced to a harmless alphabet (safeText); the model key is one of five.
% The numerical kernels are copied from opendpd.runtime as local functions, so the generated class and opendpd.apply run
% one text.
manifest = data.Manifest;
kernel = data.Kernel;
execution = char(execution);
info = struct('key', kernel.key, 'role', safeText(manifest.run.role, 20), 'runId', safeText(manifest.run.run_id, 64), ...
    'evidence', safeText(manifest.evidence.type, 20), 'sha', checkedHash(data.SHA256), ...
    'sampleRate', manifest.signal.sample_rate_hz, 'segment', manifest.signal.nperseg, 'execution', string(execution));
body = modelBody(kernel, manifest, execution);
sources = struct();
sources.class = classSource(name, info, body);
sources.step = stepSource(name, info, execution);
sources.check = checkSource(name, info, data, execution);
sources.readme = readmeSource(name, info, body);
end

function limit = parameterLimit()
limit = 250000;          % numbers written as source text; beyond this a package should be run with opendpd.apply
end

% ----- the class ---------------------------------------------------------------------------------------------------

function text = classSource(name, info, body)
rows = [
    "classdef " + name + " < matlab.System"
    "    % " + upper(name) + "  " + info.key + " " + info.role + " from OpenDPD, executed as " + info.execution + "."
    "    %   " + codegenMark() + " from an opendpd-model-v1 package. Do not edit it:"
    "    %   generate it again. It needs no OpenDPD toolbox, no Python and no data file, and it is written in the subset that"
    "    %   MATLAB Coder and the Simulink MATLAB System block accept."
    "    %   Package SHA-256 " + info.sha + "; run " + info.runId + " (" + info.role + "); evidence: " + info.evidence + "."
    "    %   y = obj(x) takes a vector of I/Q samples (single or double, real or complex), rounds it to single as OpenDPD's"
    "    %   evaluator does, and returns a complex single column vector. Samples are not normalised or aligned: use the"
    "    %   training dataset's units and sample rate (" + numberText(info.sampleRate) + " Hz)."
    "    %   " + body.sentence
    "    properties (Constant, Hidden)"
    "        ModelKey = '" + info.key + "'"
    "        Role = '" + info.role + "'"
    "        RunId = '" + info.runId + "'"
    "        Execution = '" + info.execution + "'"
    "        SampleRateHz = " + numberText(info.sampleRate)
    "        SegmentSamples = " + countText(info.segment)
    "        PackageSHA256 = '" + info.sha + "'"
    "    end"
    "    properties (Constant, Access = private)"
    body.constants
    "    end"];
if ~isempty(body.state)
    rows = [rows
        "    properties (DiscreteState)"
        "        " + body.state.name
        "    end"];
end
rows = [rows; "    methods (Access = protected)"];
if ~isempty(body.state)
    rows = [rows
        "        function setupImpl(obj)"
        "            obj." + body.state.name + " = " + body.state.initial + ";"
        "        end"
        ""
        "        function resetImpl(obj)"
        "            obj." + body.state.name + " = " + body.state.initial + ";"
        "        end"
        ""
        "        function [sz, type, complexity] = getDiscreteStateSpecificationImpl(~, ~)"
        "            sz = " + body.state.size + ";"
        "            type = 'double';"
        "            complexity = " + body.state.complex + ";"
        "        end"
        ""];
end
rows = [rows
    body.step
    ""
    "        function validateInputsImpl(~, x)"
    "            validateattributes(x, {'single', 'double'}, {'vector', 'nonempty', 'finite', 'nonsparse'});"
    "        end"
    ""
    "        function flag = isInputSizeMutableImpl(~, ~)"
    "            flag = true;"
    "        end"
    ""
    "        function flag = isInputComplexityMutableImpl(~, ~)"
    "            flag = true;"
    "        end"
    ""
    "        function sz = getOutputSizeImpl(obj)"
    "            sz = [prod(propagatedInputSize(obj, 1)), 1];"
    "        end"
    ""
    "        function type = getOutputDataTypeImpl(~)"
    "            type = 'single';"
    "        end"
    ""
    "        function flag = isOutputComplexImpl(~)"
    "            flag = true;"
    "        end"
    ""
    "        function flag = isOutputFixedSizeImpl(obj)"
    "            flag = propagatedInputFixedSize(obj, 1);"
    "        end"
    ""
    "        function icon = getIconImpl(~)"
    "            icon = 'OpenDPD " + info.key + " " + info.role + "';"
    "        end"
    "    end"
    "    methods (Access = private)"
    body.methods
    "    end"
    "end"
    ""
    kernelFunctions(body.kernels)];
text = joinLines(rows);
end

function rows = kernelFunctions(names)
% The local functions of the generated class: the named kernels of opendpd.runtime and what they call, verbatim except
% that the package qualifier is dropped.
folder = fullfile(fileparts(mfilename('fullpath')), '..', '+runtime');
pending = string(names(:)).';          % rows, so that pending(end+1) and pending(1) = [] stay one-dimensional
done = strings(1, 0);
while ~isempty(pending)
    current = pending(1);
    pending(1) = [];
    if any(done == current)
        continue
    end
    source = fileread(fullfile(folder, current + ".m"));
    tokens = regexp(source, 'opendpd\.runtime\.(\w+)', 'tokens');
    for i = 1:numel(tokens)
        dependency = string(tokens{i}{1});
        if ~any(done == dependency) && ~any(pending == dependency)
            pending(end+1) = dependency; %#ok<AGROW>
        end
    end
    done(end+1) = current; %#ok<AGROW>
end
rows = strings(0, 1);
for current = sort(done)
    source = string(fileread(fullfile(folder, current + ".m")));
    source = replace(source, "opendpd.runtime.", "");
    source = replace(source, sprintf('\r\n'), newline);
    source = regexprep(source, '[ \t]+(?=\n)', '');
    lines = splitlines(source);
    while ~isempty(lines) && strlength(lines(end)) == 0
        lines(end) = [];
    end
    rows = [rows; lines; ""]; %#ok<AGROW>
end
end

% ----- per model --------------------------------------------------------------------------------------------------

function body = modelBody(k, manifest, execution)
streaming = strcmp(execution, 'streaming_stateful');
if streaming && ~manifest.execution.streaming_stateful.available
    error('opendpd:NoStreamingVariant', '''%s'' has no registered streaming variant; use Execution="offline_segmented".', k.key);
end
body = struct('constants', strings(0, 1), 'state', [], 'methods', strings(0, 1), 'step', strings(0, 1), ...
    'kernels', {{}}, 'sentence', "", 'count', 0);
switch k.key
    case 'mp_ls'
        body.constants = constant('Coefficients', complexColumn(k.coefficients));
        body.count = 2 * numel(k.coefficients);
        body.methods = method("segmentForward", "x", "y", ...
            "y = mpForward(x, obj.Coefficients, " + countText(k.K) + ", " + countText(k.Q) + ");");
        body.kernels = {'mpForward'};
        body.sentence = "Memory polynomial, " + countText(k.K) + " envelope orders, " + countText(k.Q) + " lags.";
    case 'gmp_ls'
        names = ["Ka", "La", "Kb", "Lb", "Mb", "Kc", "Lc", "Mc"];
        fields = strings(1, numel(names));
        for i = 1:numel(names)
            fields(i) = "'" + names(i) + "', " + countText(k.params.(names(i)));
        end
        body.constants = [constant('Coefficients', complexColumn(k.coefficients))
            "        Terms = struct(" + join(fields, ", ") + ");"];
        body.count = 2 * numel(k.coefficients);
        body.methods = method("segmentForward", "x", "y", "y = gmpPolynomialForward(x, obj.Coefficients, obj.Terms);");
        body.kernels = {'gmpPolynomialForward'};
        body.sentence = "Generalised memory polynomial (least squares).";
    case 'gmp'
        body.constants = constant('Weight', realColumn(k.weight));
        body.count = numel(k.weight);
        call = "gmpForward(%s, obj.Weight, " + countText(k.M) + ", " + countText(k.D) + ")";
        body.kernels = {'gmpForward'};
        body.sentence = "Gradient-trained GMP, memory length " + countText(k.M) + ", degree " + countText(k.D) + ".";
        keep = 2 * (k.M - 1);
        if streaming && keep > 0
            body.state = struct('name', "InputHistory", 'initial', "complex(zeros(" + countText(keep) + ", 1))", ...
                'size', "[" + countText(keep) + " 1]", 'complex', "true");
            body.methods = method("streamForward", "chunk, history", "[y, history]", [
                "block = [history; chunk];"
                "full = " + replace(call, "%s", "block") + ";"
                "y = full(" + countText(keep) + " + 1:end);"
                "history = block(numel(block) - " + countText(keep) + " + (1:" + countText(keep) + "));"]);
        else
            body.methods = method("segmentForward", "x", "y", "y = " + replace(call, "%s", "x") + ";");
        end
    case {'gru', 'tres_gru'}
        body = recurrentBody(body, k, streaming);
    otherwise
        error('opendpd:Package', 'Model "%s" is not supported by opendpd.generateCode.', k.key);
end
if body.count > parameterLimit()
    error('opendpd:CodegenTooLarge', ['%d numbers is more than opendpd.generateCode writes as source text (%d); ' ...
        'run the package with opendpd.apply instead.'], body.count, parameterLimit());
end
if streaming
    if isempty(body.state)
        body.step = stepRows("obj.segmentForward(signal)", []);
    else
        body.step = stepRows("obj.streamForward(signal, obj." + body.state.name + ")", body.state.name);
    end
    body.sentence = body.sentence + " Execution streaming_stateful: reset(obj) returns the state to zero; chunk " + ...
        "boundaries do not change the result.";
else
    body.step = offlineStepRows();
    body.sentence = body.sentence + " Execution offline_segmented: the waveform is cut into segments of SegmentSamples, " + ...
        "the state is zero at the start of each, the last one is zero padded and trimmed again; nothing is carried " + ...
        "from one call to the next.";
end
end

function body = recurrentBody(body, k, streaming)
tres = strcmp(k.key, 'tres_gru');
layers = numel(k.weightIH);
hidden = size(k.weightHH{1}, 2);
constants = strings(0, 1);
count = 0;
for l = 1:layers
    constants = [constants
        constant("WeightIH" + l, matrixText(k.weightIH{l}))
        constant("WeightHH" + l, matrixText(k.weightHH{l}))
        constant("BiasIH" + l, realColumn(k.biasIH{l}))
        constant("BiasHH" + l, realColumn(k.biasHH{l}))]; %#ok<AGROW>
    count = count + numel(k.weightIH{l}) + numel(k.weightHH{l}) + numel(k.biasIH{l}) + numel(k.biasHH{l});
end
constants = [constants; constant('FcWeight', matrixText(k.fcWeight))];
count = count + numel(k.fcWeight);
if tres
    constants = [constants; constant('Conv1', arrayText(k.conv1)); constant('Conv2', arrayText(k.conv2))];
    count = count + numel(k.conv1) + numel(k.conv2);
    body.kernels = {'gruLayer', 'tresFeatures', 'tresSkip'};
    body.sentence = "TRes-GRU, " + countText(layers) + " layer(s) of " + countText(hidden) + ...
        " units, bias-free head, TCN skip path.";
else
    constants = [constants; constant('FcBias', realColumn(k.fcBias))];
    count = count + numel(k.fcBias);
    body.kernels = {'gruLayer'};
    body.sentence = "GRU, " + countText(layers) + " layer(s) of " + countText(hidden) + " units, linear head.";
end
body.constants = constants;
body.count = count;
calls = strings(0, 1);
previous = "features";
for l = 1:layers
    if streaming
        start = "hidden(:, " + l + ")";
        final = "state" + l;
    else
        start = "zeros(" + countText(hidden) + ", 1)";
        final = "~";
    end
    calls = [calls
        "[out" + l + ", " + final + "] = gruLayer(" + previous + ", obj.WeightIH" + l + ", obj.WeightHH" + l + ...
        ", obj.BiasIH" + l + ", obj.BiasHH" + l + ", " + start + ");"]; %#ok<AGROW>
    previous = "out" + l;
end
if tres
    lines = ["iq = [real(x), imag(x)];"
        "skip = tresSkip(iq, obj.Conv1, obj.Conv2);"
        "features = tresFeatures(iq);"
        calls
        "out = " + previous + " * obj.FcWeight.' + skip;"
        "y = complex(out(:, 1), out(:, 2));"];
    body.methods = method("segmentForward", "x", "y", lines);
elseif streaming
    lines = ["features = [real(x), imag(x)];"
        calls
        "hidden = [" + join("state" + (1:layers), ", ") + "];"
        "out = " + previous + " * obj.FcWeight.' + obj.FcBias.';"
        "y = complex(out(:, 1), out(:, 2));"];
    body.methods = method("streamForward", "x, hidden", "[y, hidden]", lines);
    body.state = struct('name', "RecurrentState", 'initial', "zeros(" + countText(hidden) + ", " + countText(layers) + ")", ...
        'size', "[" + countText(hidden) + " " + countText(layers) + "]", 'complex', "false");
else
    lines = ["features = [real(x), imag(x)];"
        calls
        "out = " + previous + " * obj.FcWeight.' + obj.FcBias.';"
        "y = complex(out(:, 1), out(:, 2));"];
    body.methods = method("segmentForward", "x", "y", lines);
end
end

function rows = offlineStepRows()
rows = [
    "        function y = stepImpl(obj, x)"
    "            signal = complex(double(single(real(x(:)))), double(single(imag(x(:)))));"
    "            count = numel(signal);"
    "            segment = obj.SegmentSamples;"
    "            segments = ceil(count / segment);"
    "            padded = [signal; complex(zeros(segments * segment - count, 1))];"
    "            out = complex(zeros(segments * segment, 1));"
    "            for s = 1:segments"
    "                range = (s - 1) * segment + (1:segment);"
    "                out(range) = obj.segmentForward(padded(range));"
    "            end"
    "            y = complex(single(real(out(1:count))), single(imag(out(1:count))));"
    "        end"];
end

function rows = stepRows(call, stateName)
rows = [
    "        function y = stepImpl(obj, x)"
    "            signal = complex(double(single(real(x(:)))), double(single(imag(x(:)))));"];
if isempty(stateName)
    rows = [rows; "            out = " + call + ";"];
else
    rows = [rows
        "            [out, next] = " + call + ";"
        "            obj." + stateName + " = next;"];
end
rows = [rows
    "            y = complex(single(real(out)), single(imag(out)));"
    "        end"];
end

function rows = method(name, inputs, outputs, lines)
rows = [
    "        function " + outputs + " = " + name + "(obj, " + inputs + ")"
    "            " + string(lines(:))
    "        end"];
end

function rows = constant(name, value)
% "Name = literal;" where the literal may take several lines.
value = string(value(:));
rows = value;
rows(1) = "        " + name + " = " + value(1);
rows(end) = rows(end) + ";";
end

% ----- the entry point, the check, the readme ---------------------------------------------------------------------

function text = stepSource(name, info, execution)
if strcmp(execution, 'offline_segmented')
    summary = "the waveform x through " + name + "; nothing is carried from one call to the next.";
else
    summary = "x through " + name + ", with its state kept from one call to the next.";
end
rows = [
    "function y = " + name + "Step(x) %#codegen"
    "%" + upper(name) + "STEP Entry point for MATLAB Coder: " + summary
    "%   codegen " + name + "Step -args {coder.typeof(complex(single(0)), [Inf 1])}"
    "%   " + codegenMark() + " for the package with SHA-256 " + info.sha + "."
    "persistent model"
    "if isempty(model)"
    "    model = " + name + ";"
    "end"
    "y = step(model, x);"
    "end"
    ""];
text = joinLines(rows);
end

function text = checkSource(name, info, data, execution)
g = data.Golden;
manifest = data.Manifest;
streaming = strcmp(execution, 'streaming_stateful');
outputName = 'output_offline_segmented';
if streaming
    outputName = 'output_streaming_stateful';
end
if ~isfield(g, 'input') || ~isfield(g, outputName)
    error('opendpd:Package', 'This package has no golden %s vector.', outputName);
end
samples = size(g.input, 1);
if samples > 50000 || size(g.input, 2) ~= 2 || ~isequal(size(g.(outputName)), size(g.input))
    error('opendpd:CodegenTooLarge', ...
        'The golden vector must be N-by-2 with N at most 50000 to be written into a check function.');
end
tolerance = min(double(manifest.golden.tolerance_abs), opendpd.Model.MaxGoldenTolerance);
rows = [
    "function report = " + name + "Check()"
    "%" + upper(name) + "CHECK Run " + name + " on the golden test vector of the package it was generated from."
    "%   report = " + name + "Check() returns model, execution, samples, tolerance_abs, max_abs_error and passed. The vector"
    "%   is OpenDPD's own output (" + info.execution + ") for a seeded noise input of " + countText(samples) + " samples; a pass shows that"
    "%   this MATLAB release computes the model the way OpenDPD does. It does not show where the package came from:"
    "%   compare its SHA-256 (" + info.sha + ") with the value you were given."
    "%   " + codegenMark() + " for that package."
    "input = " + goldenText(g.input) + ";"
    "expected = " + goldenText(g.(outputName)) + ";"
    "tolerance = " + numberText(tolerance) + ";"
    "x = complex(input(:, 1), input(:, 2));"
    "model = " + name + ";"];
if streaming
    rows = [rows
        "chunk = " + countText(manifest.golden.streaming_chunk_samples) + ";"
        "y = complex(zeros(numel(x), 1, 'single'));"
        "for first = 1:chunk:numel(x)"
        "    last = min(first + chunk - 1, numel(x));"
        "    y(first:last) = step(model, x(first:last));"
        "end"];
else
    rows = [rows; "y = step(model, x);"];
end
rows = [rows
    "difference = [real(double(y)) - double(expected(:, 1)); imag(double(y)) - double(expected(:, 2))];"
    "if numel(y) ~= size(expected, 1) || any(~isfinite(difference))"
    "    worst = Inf;"
    "else"
    "    worst = max(abs(difference));"
    "end"
    "report = struct('model', '" + info.key + "', 'execution', '" + info.execution + "', 'samples', numel(x), ..."
    "    'tolerance_abs', tolerance, 'max_abs_error', worst, 'passed', worst <= tolerance);"
    "end"
    ""];
text = joinLines(rows);
end

function text = readmeSource(name, info, body)
rows = [
    "# " + name
    ""
    "A standalone MATLAB System object. " + codegenMark() + " from an `opendpd-model-v1` package."
    "It needs no OpenDPD toolbox, no Python and no data file."
    ""
    "| | |"
    "|---|---|"
    "| Model | `" + info.key + "` (" + info.role + ") |"
    "| Execution | `" + info.execution + "` |"
    "| Run | `" + info.runId + "` |"
    "| Package SHA-256 | `" + info.sha + "` |"
    "| Evidence | " + info.evidence + " (model inference, not a hardware measurement) |"
    "| Sample rate of the training data | " + numberText(info.sampleRate) + " Hz |"
    "| Segment length | " + countText(info.segment) + " samples |"
    ""
    body.sentence
    ""
    "## Files"
    ""
    "* `" + name + ".m` - the System object: `obj = " + name + "; y = obj(x);` with `x` a vector of I/Q samples."
    "* `" + name + "Step.m` - the entry point for MATLAB Coder."
    "* `" + name + "Check.m` - runs the object on the golden test vector of the package: `" + name + "Check()`."
    ""
    "## Use"
    ""
    "```matlab"
    "addpath(pwd)                  % this folder"
    name + "Check()"
    "obj = " + name + ";"
    "y = obj(x);                   % x in the units and at the sample rate of the training data"
    "codegen " + name + "Step -args {coder.typeof(complex(single(0)), [Inf 1])}"
    "```"
    ""
    "In Simulink, add a MATLAB System block and set its System object name to `" + name + "`."
    ""
    "## Limits"
    ""
    "* The arithmetic is double precision on single-rounded inputs, as in the OpenDPD MATLAB runtime. It is not a"
    "  fixed-point or HDL-ready design."
    "* Samples are not normalised or aligned. The model was trained at " + numberText(info.sampleRate) + " Hz in the units of"
    "  its dataset, and it extrapolates outside the amplitude range it was trained on."
    "* The golden check shows that the class computes the model as OpenDPD did. It does not show where the package came"
    "  from: compare the SHA-256 above with the value you were given."
    ""];
text = joinLines(rows);
end

% ----- literals and text -----------------------------------------------------------------------------------------

function text = joinLines(rows)
text = char(join(string(rows(:)), newline) + newline);
end

function text = goldenText(values)
if isa(values, 'single')
    text = "single(" + join(matrixText(double(values), 'single'), newline) + ")";
else
    text = join(matrixText(double(values), 'double'), newline);
end
end

function rows = realColumn(values)
rows = matrixText(reshape(double(values), [], 1));
end

function rows = complexColumn(values)
rows = splitlines("complex(" + join(realColumn(real(values)), newline) + ", " + join(realColumn(imag(values)), newline) + ")");
end

function rows = arrayText(values)
if ndims(values) <= 2
    rows = matrixText(double(values));
    return
end
vector = join(matrixText(reshape(double(values), [], 1)), newline);
rows = splitlines("reshape(" + vector + ", [" + join(string(size(values)), " ") + "])");
end

function rows = matrixText(values, kind)
% A numeric matrix as a MATLAB literal on several lines. A column is written four numbers to a line separated by ;
% a matrix one row (of eight numbers to a line, continued with ...) at a time with ; between the rows. Each number is the
% shortest decimal text that reads back as exactly the stored value (as a double, or as a single for KIND "single").
if nargin < 2
    kind = 'double';
end
values = double(values);
if ~all(isfinite(values(:)))
    error('opendpd:Package', 'A weight is not finite; opendpd.generateCode writes finite numbers only.');
end
[r, c] = size(values);
rows = strings(0, 1);
if c == 1
    perLine = 4;
    pieces = numbersText(values(:, 1), kind);
    for first = 1:perLine:r
        rows(end+1, 1) = join(pieces(first:min(first + perLine - 1, r)), "; ") + "; ..."; %#ok<AGROW>
    end
    rows(end) = replace(rows(end), "; ...", "");
else
    perLine = 8;
    for i = 1:r
        pieces = numbersText(values(i, :), kind);
        for first = 1:perLine:c
            last = min(first + perLine - 1, c);
            line = join(pieces(first:last), ", ");
            if last < c
                line = line + ", ...";
            elseif i < r
                line = line + "; ...";
            end
            rows(end+1, 1) = line; %#ok<AGROW>
        end
    end
end
rows = "            " + rows;
rows(1) = "[ ..." + newline + rows(1);
rows(end) = rows(end) + "]";
rows = splitlines(join(rows, newline));
end

function pieces = numbersText(values, kind)
pieces = strings(1, numel(values));
for i = 1:numel(values)
    pieces(i) = shortest(values(i), kind);
end
end

function text = shortest(value, kind)
% The shortest decimal text that reads back as exactly VALUE: as a double (15 to 17 digits) or, for kind 'single', as the
% same single (6 to 9 digits).
if strcmp(kind, 'single')
    digits = 6:9;
else
    digits = 15:17;
end
for d = digits
    text = string(sprintf('%.*g', d, value));
    back = str2double(text);
    if (strcmp(kind, 'single') && single(back) == single(value)) || (~strcmp(kind, 'single') && back == value)
        return
    end
end
end

function text = numberText(value)
if ~(isnumeric(value) && isscalar(value) && isfinite(value))
    error('opendpd:Package', 'A number written into the code must be finite.');
end
text = shortest(double(value), 'double');
end

function text = countText(value)
if ~(isnumeric(value) && isscalar(value) && isfinite(value) && value == fix(value) && value >= 0 && value <= 2^31)
    error('opendpd:Package', 'A count written into the code must be a non-negative integer.');
end
text = string(sprintf('%d', double(value)));
end

function text = safeText(value, limit)
% Text from the manifest goes into comments and quoted constants only after it has been reduced to a harmless alphabet:
% no quote, no percent sign, no line break, nothing that could end a comment or a string and start code.
value = reshape(char(value), 1, []);
value(~ismember(value, ['a':'z' 'A':'Z' '0':'9' ' _.:/@+=,()-'])) = '?';
if numel(value) > limit
    value = [value(1:limit) '...'];
end
text = string(value);
end

function hash = checkedHash(value)
if isempty(regexp(value, '^[0-9a-f]{64}$', 'once'))
    error('opendpd:Package', 'The package hash is not a SHA-256.');
end
hash = string(value);
end

function mark = codegenMark()
mark = opendpd.internal.codegenMark();
end
