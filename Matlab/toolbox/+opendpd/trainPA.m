function job = trainPA(project, dataset, options)
%TRAINPA Submit PA training. Training/ModelParameters use the OpenDPD schema.
arguments
    project (1,1) opendpd.Project
    dataset
    options.Model (1,1) string = "gru"
    options.ModelParameters (1,1) struct = struct()
    options.Training (1,1) struct = struct()
    options.Device (1,1) string {mustBeMember(options.Device, ["cpu", "cuda", "mps"])} = "cpu"
    options.NumThreads (1,1) double {mustBeInteger, mustBePositive} = 1
    options.Profile (1,1) string = "opendpd-spectral-v2"
end
job = opendpd.submit(project, trainingConfig('train_pa', dataset, options));
end
