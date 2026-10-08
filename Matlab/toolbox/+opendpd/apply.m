function [y, info] = apply(target, x, options)
%APPLY Apply a succeeded PA or DPD run, or a loaded model package, to a waveform; returns a complex single column vector.
% target is an opendpd.Job (run in the Python SDK, on CPU) or an opendpd.Model from opendpd.load (plain MATLAB, no Python).
% Execution="offline_segmented" (default) is how the run was scored: state resets at the run's stored
% segment length and the last segment is zero padded. Execution="streaming_stateful" (alias "streaming")
% carries one state across chunks of ChunkSamples and exists only for models with a registered streaming
% variant (gru, gmp); its output is not the signal the stored report scored. info.execution, info.limitations
% and, for streaming, info.streaming say what produced y. No normalization is applied. A DPD output is the
% predistorted PA input, before a physical or simulated PA.
arguments
    target (1,1) {mustBeA(target, ["opendpd.Job", "opendpd.Model"])}
    x {mustBeNumeric}
    options.Execution (1,1) string {mustBeMember(options.Execution, ...
        ["offline_segmented", "streaming_stateful", "streaming"])} = "offline_segmented"
    options.ChunkSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.Timeout (1,1) double {mustBePositive, mustBeFinite} = 120
end
if isa(target, 'opendpd.Model')
    [y, info] = target.apply(x, Execution=options.Execution, ChunkSamples=options.ChunkSamples);
    return
end
module = bridge();
output = module.apply(target.Backend, asPythonIQ(x), char(options.Execution), options.Timeout, options.ChunkSamples);
iq = single(output{1});
y = complex(iq(:,1), iq(:,2));
info = jsondecode(char(output{2}));
end
