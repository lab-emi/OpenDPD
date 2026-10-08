function dataset = importIQ(project, x, y, options)
%IMPORTIQ Import paired PA input/output vectors without normalization.
% Supply sample rate and occupied bandwidth in Hz. Name is a unique dataset ID.
arguments
    project (1,1) opendpd.Project
    x {mustBeNumeric}
    y {mustBeNumeric}
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
    options.Bandwidth (1,1) double {mustBePositive, mustBeFinite}
    options.Name (1,1) string = ""
    options.SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)} = 256
    options.Subchannels (1,1) double {mustBeInteger, mustBePositive} = 1
    options.GuardSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 256
    options.Origin (1,1) string {mustBeMember(options.Origin, ["unknown", "measured", "synthetic"])} = "unknown"
    options.AmplitudeUnits (1,1) string {mustBeMember(options.AmplitudeUnits, ["unknown", "normalized", "volts"])} = "unknown"
end
config = importOptions(options);
module = bridge();
dataset = jsondecode(char(module.import_iq(project.Backend, asPythonIQ(x), asPythonIQ(y), ...
    jsonencode(config))));
end
