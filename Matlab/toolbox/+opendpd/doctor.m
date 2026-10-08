function info = doctor()
%DOCTOR Check the selected Python environment and report preview capabilities.
module = bridge();
info = jsondecode(char(module.diagnostics()));
info.matlab_release = version('-release');
environment = pyenv;
info.python_execution_mode = char(environment.ExecutionMode);
end
