function dataset = importIQ(project, x, y, options)
%IMPORTIQ Import paired PA input/output vectors without normalization.
% Supply SampleRate and Bandwidth in Hz, and SegmentSamples. Name is a unique dataset ID.
% SegmentSamples has no default on purpose: it is the PSD segment length of every spectral metric and the
% interval at which evaluation restarts a model's state. Studio's generated signals use 512-4096.
arguments
    project (1,1) opendpd.Project
    x {mustBeNumeric}
    y {mustBeNumeric}
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
    options.Bandwidth (1,1) double {mustBePositive, mustBeFinite}
    options.SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)}
    options.Name (1,1) string = ""
    options.Subchannels (1,1) double {mustBeInteger, mustBePositive} = 1
    options.GuardSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 256
    options.Origin (1,1) string {mustBeMember(options.Origin, ["unknown", "measured", "synthetic"])} = "unknown"
    options.AmplitudeUnits (1,1) string {mustBeMember(options.AmplitudeUnits, ["unknown", "normalized", "volts"])} = "unknown"
    options.Source (1,1) struct = struct()
end
config = importOptions(options);
module = bridge();
dataset = jsondecode(char(module.import_iq(project.Backend, asPythonIQ(x), asPythonIQ(y), ...
    jsonencode(config))));
end
