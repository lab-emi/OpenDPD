function [y, info] = apply(job, x, options)
%APPLY Apply a trained GRU on CPU; returns a complex single column vector.
% State resets at the stored segment length. No normalization is applied.
% DPD output is the predistorted PA input, before a physical or simulated PA.
arguments
    job (1,1) opendpd.Job
    x {mustBeNumeric}
    options.Execution (1,1) string {mustBeMember(options.Execution, "offline_segmented")} = "offline_segmented"
    options.Timeout (1,1) double {mustBePositive, mustBeFinite} = 120
end
module = bridge();
output = module.apply(job.Backend, asPythonIQ(x), char(options.Execution), options.Timeout);
iq = single(output{1});
y = complex(iq(:,1), iq(:,2));
info = jsondecode(char(output{2}));
end
