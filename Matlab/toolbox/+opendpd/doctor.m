function info = doctor()
%DOCTOR Check the selected Python environment and report what apply supports.
% info.apply_models and info.apply_execution come from the installed OpenDPD; info.apply_streaming_models
% lists the models that can run with Execution="streaming".
module = bridge();
info = jsondecode(char(module.diagnostics()));
info.matlab_release = version('-release');
environment = pyenv;
info.python_execution_mode = char(environment.ExecutionMode);
end
