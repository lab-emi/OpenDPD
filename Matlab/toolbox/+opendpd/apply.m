function [y, info] = apply(target, x, options)
%APPLY Apply a succeeded PA or DPD run, or a loaded model package, to a waveform; returns a complex single column vector.
% target is an opendpd.Job (run in the Python SDK, on CPU), an opendpd.Model from opendpd.load (plain MATLAB, no Python) or an
% opendpd.FixedModel (a fixed-point-v1 package, plain MATLAB integers). Execution="auto" (the default for Job and Model)
% runs a model that has a registered streaming variant (gru, gmp) as "streaming_stateful" and every other model as
% "offline_segmented". Execution="offline_segmented" is how the run was scored: state resets at the run's stored segment
% length and the last segment is zero padded. Execution="streaming_stateful" (alias "streaming") carries one state across
% chunks of ChunkSamples and exists only for models with a registered streaming variant; its output is not the signal the
% stored report scored. A FixedModel has that streaming execution only, so it is its default. info.execution says which
% semantics ran, info.execution_requested what was asked for, info.execution_reason why (for "auto"), and info.limitations
% and, for streaming, info.streaming what that means. No normalization is applied (a FixedModel quantises the input to its
% input format and saturates beyond it). A DPD output is the predistorted PA input, before a physical or simulated PA.
arguments
    target (1,1) {mustBeA(target, ["opendpd.Job", "opendpd.Model", "opendpd.FixedModel"])}
    x {mustBeNumeric}
    options.Execution (1,1) string {mustBeMember(options.Execution, ...
        ["", "auto", "offline_segmented", "streaming_stateful", "streaming"])} = ""
    options.ChunkSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.Timeout (1,1) double {mustBePositive, mustBeFinite} = 120
end
if isa(target, 'opendpd.FixedModel')
    [y, info] = target.apply(x, Execution=options.Execution, ChunkSamples=options.ChunkSamples);
    return
end
execution = options.Execution;
if execution == ""
    execution = "auto";
end
if isa(target, 'opendpd.Model')
    [y, info] = target.apply(x, Execution=execution, ChunkSamples=options.ChunkSamples);
    return
end
module = bridge();
output = module.apply(target.Backend, asPythonIQ(x), char(execution), options.Timeout, options.ChunkSamples);
iq = single(output{1});
y = complex(iq(:,1), iq(:,2));
info = jsondecode(char(output{2}));
end
