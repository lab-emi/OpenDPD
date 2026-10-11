function job = getRun(project, runID)
%GETRUN Reconnect to a stored job using its durable run ID.
arguments
    project (1,1) opendpd.Project
    runID (1,1) string
end
job = opendpd.Job(project, project.Backend.job(char(runID)));
end
