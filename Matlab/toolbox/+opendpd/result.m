function report = result(job)
%RESULT Read the stored evaluation result, including profile and evidence type.
arguments
    job (1,1) opendpd.Job
end
module = bridge();
report = jsondecode(char(module.encode(job.Backend.result())));
end
