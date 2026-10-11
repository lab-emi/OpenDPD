function record = cancel(job)
%CANCEL Request cancellation; status remains cancel_requested until stopped.
arguments
    job (1,1) opendpd.Job
end
module = bridge();
record = jsondecode(char(module.encode(job.Backend.cancel())));
end
