function summary = export(job, file, options)
%EXPORT Write a succeeded run as an opendpd-model-v1 package (weights, manifest, golden test vector; no code).
% summary = opendpd.export(job, "apa-dpd.opendpd.zip") needs the Python SDK (the run lives in a workspace); the
% resulting package does not: load it anywhere with opendpd.load. Supported models: gru, tres_gru, gmp, mp_ls, gmp_ls
% (the models opendpd.apply supports). The same run always gives the same bytes. The golden test input is
% synthetic noise with the training input's amplitude statistics, never a slice of your data.
arguments
    job (1,1) opendpd.Job
    file (1,1) string
    options.Timeout (1,1) double {mustBePositive, mustBeFinite} = 300
end
module = opendpd.internal.bridge();
summary = jsondecode(char(module.export_model(job.Backend, char(file), options.Timeout)));
end
