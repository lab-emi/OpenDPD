function dataset = importMAT(project, filename, options)
%IMPORTMAT Import named numeric PA input/output variables from a MAT file of any version.
% MATLAB reads the file, so v7.3 (HDF5) works as well as v7. Each variable is a single/double vector (real
% vectors are I-only signals) or a real N-by-2 I/Q matrix. SegmentSamples has no default; see importIQ.
% The dataset records the file name, its SHA-256 and the variable names.
arguments
    project (1,1) opendpd.Project
    filename (1,1) string {mustBeFile}
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
    options.Bandwidth (1,1) double {mustBePositive, mustBeFinite}
    options.SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)}
    options.InputVariable (1,1) string = "x"
    options.OutputVariable (1,1) string = "y"
    options.Name (1,1) string = ""
    options.Subchannels (1,1) double {mustBeInteger, mustBePositive} = 1
    options.GuardSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 256
    options.Origin (1,1) string {mustBeMember(options.Origin, ["unknown", "measured", "synthetic"])} = "unknown"
    options.AmplitudeUnits (1,1) string {mustBeMember(options.AmplitudeUnits, ["unknown", "normalized", "volts"])} = "unknown"
end
[ok, attributes] = fileattrib(filename);
if ~ok
    error('opendpd:MATFile', 'Cannot resolve MAT file: %s', filename);
end
path = attributes.Name;
names = [options.InputVariable, options.OutputVariable];
if names(1) == names(2)
    error('opendpd:MATFile', 'Input and output variables must be different.');
end
listed = whos('-file', path);
signals = cell(1, 2);
roles = ["input", "output"];
record = struct();
for k = 1:2
    item = listed(strcmp({listed.name}, char(names(k))));
    if isempty(item) || ~any(strcmp(item.class, {'single', 'double'})) || item.sparse
        error('opendpd:MATFile', ['MAT variable ''%s'' must exist and be a full single/double numeric array. ' ...
            'Cells, structs, sparse arrays and integer types are not imported.'], names(k));
    end
    value = load(path, char(names(k)));
    signals{k} = matSignal(value.(char(names(k))), names(k));
    record.(roles(k)) = struct('variable', char(names(k)), 'class', item.class, ...
        'complex', logical(item.complex), 'size', double(item.size));
end
[~, base, extension] = fileparts(path);
module = bridge();
source = struct('format', 'mat', 'name', [base extension], 'sha256', char(module.file_sha256(path)), ...
    'input_variable', char(options.InputVariable), 'output_variable', char(options.OutputVariable), ...
    'variables', record);
forwarded = rmfield(options, {'InputVariable', 'OutputVariable'});
arguments = namedargs2cell(forwarded);
dataset = opendpd.importIQ(project, signals{1}, signals{2}, arguments{:}, Source=source);
end

function z = matSignal(value, name)
% A complex column vector. Vectors are sample sequences (real ones are I-only); a real N-by-2 matrix is I/Q.
if isvector(value)
    z = complex(value(:));
elseif ismatrix(value) && isreal(value) && size(value, 2) == 2
    z = complex(value(:, 1), value(:, 2));
else
    error('opendpd:MATFile', 'MAT variable ''%s'' must be a vector or a real N-by-2 I/Q matrix, not %s.', ...
        name, mat2str(size(value)));
end
end
