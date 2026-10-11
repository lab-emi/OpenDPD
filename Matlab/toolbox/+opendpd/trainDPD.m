function job = trainDPD(project, dataset, options)
%TRAINDPD Submit DPD training through a succeeded PA surrogate run.
% Device and NumThreads behave as in trainPA.
arguments
    project (1,1) opendpd.Project
    dataset
    options.PA (1,1) opendpd.Job
    options.Model (1,1) string = "gru"
    options.ModelParameters (1,1) struct = struct()
    options.Training (1,1) struct = struct()
    options.Device (1,1) string {mustBeMember(options.Device, ["auto", "cpu", "cuda", "mps"])} = "auto"
    options.DeviceIndex (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.NumThreads (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.Profile (1,1) string = "opendpd-spectral-v2"
end
if ~isfield(options, 'PA')
    error('opendpd:PARequired', 'Supply PA=paRun from a succeeded PA training job.');
end
if options.PA.Project.Workspace ~= project.Workspace
    error('opendpd:WorkspaceMismatch', 'PA must belong to this workspace.');
end
config = trainingConfig(project, 'train_dpd', dataset, options);
config.pa_reference = struct('run_id', options.PA.ID);
job = opendpd.submit(project, config);
end
