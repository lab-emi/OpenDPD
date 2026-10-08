function dataset = importMAT(project, filename, options)
%IMPORTMAT Import named numeric PA input/output variables from a MAT v7 file.
% For v7.3 files, first save the numeric variables using save(..., '-v7').
arguments
    project (1,1) opendpd.Project
    filename (1,1) string {mustBeFile}
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
    options.Bandwidth (1,1) double {mustBePositive, mustBeFinite}
    options.InputVariable (1,1) string = "x"
    options.OutputVariable (1,1) string = "y"
    options.Name (1,1) string = ""
    options.SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)} = 256
    options.Subchannels (1,1) double {mustBeInteger, mustBePositive} = 1
    options.GuardSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 256
    options.Origin (1,1) string {mustBeMember(options.Origin, ["unknown", "measured", "synthetic"])} = "unknown"
    options.AmplitudeUnits (1,1) string {mustBeMember(options.AmplitudeUnits, ["unknown", "normalized", "volts"])} = "unknown"
end
config = importOptions(options);
config.input_variable = options.InputVariable;
config.output_variable = options.OutputVariable;
% Python has its own working directory. Resolve MATLAB-relative paths here.
[ok, attributes] = fileattrib(filename);
if ~ok
    error('opendpd:MATFile', 'Cannot resolve MAT file: %s', filename);
end
module = bridge();
dataset = jsondecode(char(module.import_mat(project.Backend, attributes.Name, jsonencode(config))));
end
