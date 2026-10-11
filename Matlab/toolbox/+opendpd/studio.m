function link = studio(projectOrWorkspace, options)
%STUDIO Use OpenDPD Studio as the toolbox GUI, with a live MATLINK connection.
% link = opendpd.studio(workspace) opens Studio's MATLINK page in your browser.
% Leave MATLAB open; a timer handles named signal/report transfer requests.
% opendpd.disconnect(link) detaches MATLAB and leaves Studio jobs running.
arguments
    projectOrWorkspace = ""
    options.PythonExecutable (1,1) string = ""
    options.SourceDirectory (1,1) string = ""
    options.OpenBrowser (1,1) logical = true
    options.Label (1,1) string = "MATLAB R" + string(version('-release'))
end
owned = ~isa(projectOrWorkspace, 'opendpd.Project');
if owned
    environment = pyenv;
    mode = "OutOfProcess";
    if string(environment.Status) == "Loaded", mode = string(environment.ExecutionMode); end
    executable = options.PythonExecutable;
    source = options.SourceDirectory;
    if strlength(executable) == 0 && string(environment.Status) ~= "Loaded"
        executable = string(getpref('OpenDPDToolbox', 'Python', ''));
    end
    if strlength(source) == 0, source = string(getpref('OpenDPDToolbox', 'Source', '')); end
    opendpd.setup(PythonExecutable=executable, SourceDirectory=source, ExecutionMode=mode);
    workspace = string(projectOrWorkspace);
    if strlength(workspace) == 0
        userHome = getenv('HOME');
        if ispc, userHome = getenv('USERPROFILE'); end
        if isempty(userHome), userHome = pwd; end
        workspace = string(getpref('OpenDPDToolbox', 'Workspace', fullfile(userHome, 'opendpd-matlab-workspace')));
    end
    project = opendpd.openProject(workspace);
else
    project = projectOrWorkspace;
end
key = char(project.Workspace);
link = matlinkRegistry('get', key);
if ~isempty(link) && ~link.isCurrent()
    delete(link);
    link = [];
    if ~owned
        project = opendpd.openProject(string(key));
        owned = true;
    end
end
if isempty(link)
    try
        link = opendpd.MATLABBridge(project, OwnsConnection=owned, Label=options.Label);
        matlinkRegistry('put', key, link);
    catch cause
        if owned, opendpd.closeProject(project); end
        rethrow(cause);
    end
elseif owned
    opendpd.closeProject(project);
end
setpref('OpenDPDToolbox', 'Workspace', key);
environment = pyenv;
setpref('OpenDPDToolbox', 'Python', char(environment.Executable));
if strlength(options.SourceDirectory) > 0
    setpref('OpenDPDToolbox', 'Source', char(options.SourceDirectory));
end
if options.OpenBrowser, opendpd.openStudio(link.Project, Page="matlink"); end
if nargout == 0
    fprintf('MATLINK connected. Use the MATLINK tab in OpenDPD Studio. Leave MATLAB open.\n');
    clear link;
end
end
