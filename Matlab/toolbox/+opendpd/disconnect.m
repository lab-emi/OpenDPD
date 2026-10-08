function disconnect(target)
%DISCONNECT Detach MATLINK; Studio's service and training jobs keep running.
% No argument disconnects all MATLINK bridges in this MATLAB process.
% Pass a MATLABBridge or workspace path to disconnect one workspace.
arguments
    target = ""
end
if isa(target, 'opendpd.MATLABBridge')
    if isvalid(target), delete(target); end
    return
end
workspace = string(target);
if strlength(workspace) == 0
    links = matlinkRegistry('all');
    for index = 1:numel(links), delete(links{index}); end
else
    path = py.pathlib.Path(char(workspace));
    path = path.expanduser();
    if ~logical(path.is_absolute())
        base = py.pathlib.Path(pwd);
        path = base.joinpath(path);
    end
    path = path.resolve();
    link = matlinkRegistry('get', char(py.str(path)));
    if ~isempty(link), delete(link); end
end
end
