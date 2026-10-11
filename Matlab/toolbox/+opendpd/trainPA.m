function job = trainPA(project, dataset, options)
%TRAINPA Submit PA training. Training/ModelParameters use the OpenDPD schema.
% Device="auto" (default) picks Studio's own default: the first detected of cuda, mps, cpu; least-squares
% models always run on cpu. NumThreads=0 lets the service decide.
arguments
    project (1,1) opendpd.Project
    dataset
    options.Model (1,1) string = "gru"
    options.ModelParameters (1,1) struct = struct()
    options.Training (1,1) struct = struct()
    options.Device (1,1) string {mustBeMember(options.Device, ["auto", "cpu", "cuda", "mps"])} = "auto"
    options.DeviceIndex (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.NumThreads (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.Profile (1,1) string = "opendpd-spectral-v2"
end
job = opendpd.submit(project, trainingConfig(project, 'train_pa', dataset, options));
end
