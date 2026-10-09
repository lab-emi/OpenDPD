function module = bridge()
%BRIDGE Import the installed Python SDK without changing Python's search path.
% Shared by the toolbox's functions and its sub-packages (a private folder is invisible to +metrics).
try
    module = py.importlib.import_module('opendpd.sdk.matlab');
catch cause
    error('opendpd:PythonEnvironment', ...
        'Cannot import the OpenDPD SDK. Run opendpd.setup with the Python environment used to install this checkout.\n%s', ...
        cause.message);
end
if double(module.API_VERSION) ~= 1
    error('opendpd:SDKVersion', 'This toolbox requires OpenDPD SDK API version 1.');
end
end
