function filename = help(topic, options)
%HELP Open the toolbox's offline documentation.
% Topics: index, gui, workflow, reference, architecture, troubleshooting.
arguments
    topic (1,1) string {mustBeMember(topic, ...
        ["index", "gui", "workflow", "reference", "architecture", "troubleshooting"])} = "index"
    options.OpenBrowser (1,1) logical = true
end
root = fileparts(fileparts(mfilename('fullpath')));
filename = string(fullfile(root, 'resources', 'docs', topic + ".html"));
if ~isfile(filename)
    error('opendpd:DocumentationMissing', 'Rebuild the toolbox documentation or reinstall the toolbox.');
end
if options.OpenBrowser
    web(char(filename), '-browser');
end
if nargout == 0
    clear filename;
end
end
