function m = aclr(y, options)
%ACLR Adjacent-channel leakage ratio of a capture with an OpenDPD profile, in negative dBc (lower is better).
% m = opendpd.metrics.aclr(y, SampleRate=fs, SegmentSamples=nperseg, Waveform=w) returns m.ACLR_L, m.ACLR_R (dBc),
% Status, Reason.
% Profile "ofdm-lte20-evm-v1" (default) integrates [-9,9) MHz as the main channel and 18 MHz adjacent channels
% centred at +/-20 MHz of the Welch PSD (bins whose centre lies in a half-open band); it needs a capture rate of
% at least 58 MS/s, and it is defined for a capture of the OpenDPD LTE-20 waveform: pass the Waveform that was
% played (opendpd.waveform), otherwise every metric of the profile, ACLR included, is reported as
% "missing_reference". Profile "opendpd-spectral-v2" needs no waveform; it uses Bandwidth and Subchannels (carrier
% ACLR relative to the strongest carrier). SegmentSamples is the Welch segment length and has no default: it sets the frequency
% resolution, and the value of the ratio moves with it (by up to 0.2 dB between 1024 and 4096 samples at 122.88
% MS/s for an LTE-20 signal near -40 dBc). comm.ACPR integrates the enclosing bins of a band and differs from this
% by up to about that much; see docs/performance/matlab-parity.md.
arguments
    y {mustBeNumeric}
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
    options.SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)}
    options.Profile (1,1) string {mustBeMember(options.Profile, ["ofdm-lte20-evm-v1", "opendpd-spectral-v2"])} = "ofdm-lte20-evm-v1"
    options.Waveform (1,1) struct = struct()
    options.Bandwidth (1,1) double {mustBePositive, mustBeFinite}
    options.Subchannels (1,1) double {mustBeInteger, mustBePositive} = 1
end
if ~isfield(options, 'SampleRate')
    error('opendpd:SignalMetadata', 'SampleRate (Hz) is required.');
end
if ~isfield(options, 'SegmentSamples')
    error('opendpd:SignalMetadata', ['SegmentSamples is required: it is the Welch segment length of the PSD, which sets ' ...
        'the frequency resolution and moves the ratio. Studio''s generated signals use 512-4096.']);
end
args = namedargs2cell(options);
values = opendpd.metrics.evaluate(y, args{:});
m = struct();
status = "ok";
reason = "";
for name = ["ACLR_L", "ACLR_R"]
    row = values(values.Name == name, :);
    m.(name) = row.Value;
    if row.Status ~= "ok" && status == "ok"
        status = row.Status;
        reason = row.Reason;
    end
end
m.Status = char(status);
m.Reason = char(reason);
end
