function closeProject(project, options)
%CLOSEPROJECT Disconnect; optionally stop an SDK service and its active jobs.
% A service started by the Studio launcher must be closed in that launcher.
arguments
    project (1,1) opendpd.Project
    options.StopService (1,1) logical = false
end
project.Backend.close(pyargs('stop_service', options.StopService));
end
