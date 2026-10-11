function job = submit(project, config)
%SUBMIT Submit a complete OpenDPD ExperimentConfig struct to the shared queue.
arguments
    project (1,1) opendpd.Project
    config (1,1) struct
end
module = bridge();
job = opendpd.Job(project, module.submit(project.Backend, jsonencode(config)));
end
