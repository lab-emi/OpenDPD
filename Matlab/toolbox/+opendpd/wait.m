function completed = wait(job, options)
%WAIT Wait for success. Timeout or Ctrl+C leaves the job running.
% Call opendpd.cancel(job) explicitly to request cancellation.
arguments
    job (1,1) opendpd.Job
    options.Timeout (1,1) double {mustBePositive, mustBeFinite} = 600
    options.PollInterval (1,1) double {mustBePositive, mustBeFinite} = 0.5
end
started = tic;
while true
    record = opendpd.status(job);
    state = string(record.status);
    if state == "succeeded"
        completed = job;
        return
    end
    if any(state == ["failed", "cancelled", "interrupted"])
        detail = state;
        if isstruct(record.error) && isfield(record.error, 'message')
            detail = string(record.error.message);
        end
        error('opendpd:RunFailed', 'Run %s ended as %s: %s', job.ID, state, detail);
    end
    remaining = options.Timeout - toc(started);
    if remaining <= 0
        error('opendpd:Timeout', 'Run %s is still %s. It has not been cancelled.', job.ID, state);
    end
    pause(min(options.PollInterval, remaining));
end
end
