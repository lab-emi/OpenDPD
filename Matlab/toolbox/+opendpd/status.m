function record = status(job)
%STATUS Read task state and progress from the same queue used by Studio.
arguments
    job (1,1) opendpd.Job
end
module = bridge();
record = jsondecode(char(module.encode(job.Backend.status())));
end
