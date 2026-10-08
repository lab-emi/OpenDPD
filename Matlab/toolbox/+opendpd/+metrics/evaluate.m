function values = evaluate(y, options)
%EVALUATE Score a waveform with an OpenDPD metric profile; returns a table of every metric.
% values = opendpd.metrics.evaluate(y, SampleRate=fs, SegmentSamples=nperseg, Waveform=w) uses the code Studio and
% the run service use (opendpd.core.metrics), so the numbers equal those of a Studio result. No server is needed.
%   Profile         "ofdm-lte20-evm-v1" (default: EVM, ACLR; needs Waveform for EVM),
%                   "opendpd-spectral-v2" (carrier ACLR; needs Bandwidth, Subchannels; NMSE/IBE need Reference),
%                   "general-spectral-v1"
%   SampleRate      Hz of y
%   SegmentSamples  Welch segment length of every spectral metric; no default (it sets the frequency resolution)
%   Waveform        the struct from opendpd.waveform that was played (Seed and Subframes)
%   Reference       target signal for the reference-based metrics, same length as y
% The table has Name, Value, Unit, Status, Reason. A metric that cannot be computed has Value NaN and a Status
% (for example "missing_reference" when y does not correlate with Waveform) with the Reason; nothing is guessed.
arguments
    y {mustBeNumeric}
    options.Profile (1,1) string {mustBeMember(options.Profile, ...
        ["ofdm-lte20-evm-v1", "opendpd-spectral-v2", "general-spectral-v1"])} = "ofdm-lte20-evm-v1"
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
    options.SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)}
    options.Bandwidth (1,1) double {mustBePositive, mustBeFinite}
    options.Subchannels (1,1) double {mustBeInteger, mustBePositive} = 1
    options.Reference {mustBeNumeric} = []
    options.Waveform (1,1) struct = struct()
end
if ~isfield(options, 'SampleRate')
    error('opendpd:SignalMetadata', 'SampleRate (Hz) is required.');
end
config = struct('profile', char(options.Profile), 'sample_rate_hz', options.SampleRate, ...
    'n_sub_ch', options.Subchannels);
if isfield(options, 'SegmentSamples')
    config.nperseg = options.SegmentSamples;
end
if isfield(options, 'Bandwidth')
    config.bandwidth_hz = options.Bandwidth;
end
if isfield(options.Waveform, 'Seed') && isfield(options.Waveform, 'Subframes')
    config.waveform_seed = options.Waveform.Seed;
    config.waveform_subframes = options.Waveform.Subframes;
elseif ~isempty(fieldnames(options.Waveform))
    error('opendpd:Waveform', 'Waveform must be the struct returned by opendpd.waveform (it needs Seed and Subframes).');
end
reference = py.None;
if ~isempty(options.Reference)
    reference = opendpd.internal.asPythonIQ(options.Reference);
end
module = opendpd.internal.bridge();
rows = jsondecode(char(module.evaluate_metrics(opendpd.internal.asPythonIQ(y), reference, jsonencode(config))));
n = numel(rows);
Name = strings(n, 1);
Value = nan(n, 1);
Unit = strings(n, 1);
Status = strings(n, 1);
Reason = strings(n, 1);
for k = 1:n
    Name(k) = rows(k).name;
    if ~isempty(rows(k).value)
        Value(k) = rows(k).value;
    end
    Unit(k) = rows(k).unit;
    Status(k) = rows(k).status;
    if ~isempty(rows(k).reason)
        Reason(k) = rows(k).reason;
    end
end
values = table(Name, Value, Unit, Status, Reason);
end
