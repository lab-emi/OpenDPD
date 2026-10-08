function info = setup(options)
%SETUP Select Python; optionally use a development checkout in this session.
% Does not install packages or terminate a loaded Python session.
arguments
    options.PythonExecutable (1,1) string = ""
    options.SourceDirectory (1,1) string = ""
    options.ExecutionMode (1,1) string {mustBeMember(options.ExecutionMode, ["OutOfProcess", "InProcess"])} = "OutOfProcess"
end
if strlength(options.PythonExecutable) > 0
    pyenv(Version=options.PythonExecutable, ExecutionMode=options.ExecutionMode);
else
    pyenv(ExecutionMode=options.ExecutionMode);
end
if strlength(options.SourceDirectory) > 0
    [ok, source] = fileattrib(options.SourceDirectory);
    if ~ok || ~isfile(fullfile(source.Name, 'opendpd', 'sdk', '__init__.py'))
        error('opendpd:SourceDirectory', 'SourceDirectory must contain the OpenDPD checkout with opendpd/sdk.');
    end
    modules = py.sys.modules;
    if logical(py.operator.contains(modules, 'opendpd'))
        current = modules.get('opendpd');
        loaded = fileparts(fileparts(char(py.getattr(current, '__file__'))));
        if ~strcmp(loaded, source.Name)
            error('opendpd:SourceAlreadyLoaded', ...
                'OpenDPD is already loaded from %s. Restart Python before changing checkouts.', loaded);
        end
    end
    paths = py.sys.path;
    if double(paths.count(source.Name)) > 0
        paths.remove(source.Name);
    end
    paths.insert(int32(0), source.Name);
end
info = opendpd.doctor();
if ~info.ok
    error('opendpd:Dependencies', 'Python dependencies are unavailable: %s', strjoin(string(info.errors), '; '));
end
end
