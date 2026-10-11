classdef Project < handle
    %PROJECT Connection to a local OpenDPD workspace and its Studio service.
    % Use opendpd.openProject to connect and opendpd.closeProject to disconnect.
    properties (SetAccess = private)
        Workspace (1,1) string
    end
    properties (SetAccess = private, Hidden)
        Backend
    end
    methods
        function obj = Project(backend)
            obj.Backend = backend;
            obj.Workspace = string(py.str(backend.workspace));
        end
    end
end
