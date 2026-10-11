function config = trainingConfig(project, task, dataset, options)
device = deviceFor(project, options.Model, options.Device);
execution = struct('device', device);
if device == "cuda"
    execution.device_index = options.DeviceIndex;
end
if options.NumThreads > 0
    execution.num_threads = options.NumThreads;
end
config = struct('task', task, 'dataset', struct('id', datasetID(dataset)), ...
    'model', struct('key', options.Model, 'parameters', options.ModelParameters), ...
    'training', options.Training, 'execution', execution, ...
    'evaluation', struct('profile_id', options.Profile));
end
