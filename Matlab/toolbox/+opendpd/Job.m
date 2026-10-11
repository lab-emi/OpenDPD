classdef Job
    %JOB Durable run reference. Keep ID to reconnect with opendpd.getRun.
    properties (SetAccess = private)
        ID (1,1) string
        Project
    end
    properties (SetAccess = private, Hidden)
        Backend
    end
    methods
        function obj = Job(project, backend)
            obj.Project = project;
            obj.Backend = backend;
            obj.ID = string(backend.run_id);
        end
    end
end
