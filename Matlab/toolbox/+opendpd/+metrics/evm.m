function m = evm(y, waveform, options)
%EVM Data-aided EVM of a capture of the OpenDPD LTE-20 test waveform (ofdm-lte20-evm-v1).
% m = opendpd.metrics.evm(y, w, SampleRate=fs) with w = opendpd.waveform(Seed=..., Subframes=...) returns
%   m.EVM_RMS  percent, m.EVM_dB  dB, m.Status ("ok" or why not), m.Reason
% y is the PA output (or any capture of the played waveform) at SampleRate, which must convert to 30.72 MS/s with a
% small exact ratio. The capture is synchronised with the regenerated waveform, so any start position and a static
% complex gain and carrier offset (a few tens of Hz for a 10 ms waveform) are handled; the procedure is in
% docs/protocols/waveform-profiles.md and has not been checked against 3GPP TS 36.104 Annex E. Same code as Studio.
arguments
    y {mustBeNumeric}
    waveform (1,1) struct
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
end
if ~isfield(options, 'SampleRate')
    error('opendpd:SignalMetadata', 'SampleRate (Hz) is required.');
end
values = opendpd.metrics.evaluate(y, Profile="ofdm-lte20-evm-v1", SampleRate=options.SampleRate, Waveform=waveform);
m = pick(values, ["EVM_RMS", "EVM_DB"]);
m = renameStruct(m, "EVM_DB", "EVM_dB");
end

function m = pick(values, names)
m = struct();
status = "ok";
reason = "";
for name = names
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

function m = renameStruct(m, from, to)
m.(to) = m.(from);
m = rmfield(m, from);
end
