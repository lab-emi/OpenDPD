function project = openProject(workspace, options)
%OPENPROJECT Reuse Studio or start a local service for a workspace.
% The service and jobs continue until closeProject(..., StopService=true).
arguments
    workspace (1,1) string
    options.StartService (1,1) logical = true
    options.Timeout (1,1) double {mustBePositive, mustBeFinite} = 30
end
bridge();
sdk = py.importlib.import_module('opendpd.sdk');
path = py.pathlib.Path(char(workspace));
path = path.expanduser();
if ~logical(path.is_absolute())
    base = py.pathlib.Path(pwd);
    path = base.joinpath(path);
end
backend = sdk.open_project(path, pyargs('start', options.StartService, 'timeout', options.Timeout));
project = opendpd.Project(backend);
end
