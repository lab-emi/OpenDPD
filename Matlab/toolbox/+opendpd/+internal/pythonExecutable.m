function executable = pythonExecutable(explicit)
%PYTHONEXECUTABLE The Python that runs OpenDPD for the process transport, without loading Python into MATLAB.
% Order: the PythonExecutable option, the OPENDPD_PYTHON environment variable, the interpreter that opendpd.studio or
% opendpd.setup remembered, then the one MATLAB's pyenv is configured with (reading it does not load Python).
arguments
    explicit (1,1) string = ""
end
candidates = {char(explicit), getenv('OPENDPD_PYTHON'), char(string(getpref('OpenDPDToolbox', 'Python', ''))), configuredPyenv()};
for k = 1:numel(candidates)
    if ~isempty(candidates{k})
        if ~isfile(candidates{k})
            error('opendpd:PythonEnvironment', 'The Python executable "%s" does not exist.', candidates{k});
        end
        executable = string(candidates{k});
        return
    end
end
error('opendpd:PythonEnvironment', ['No Python executable is known. Pass PythonExecutable=..., set OPENDPD_PYTHON, or run ' ...
    'opendpd.studio/opendpd.setup once with the Python environment where OpenDPD is installed.']);
end

function path = configuredPyenv()
try
    path = char(pyenv().Executable);
catch
    path = '';
end
end
