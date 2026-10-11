function url = openStudio(project, options)
%OPENSTUDIO Open a page in the shared Studio web application.
% Page: home, matlink, datasets, experiments, new-experiment, results, run, result.
% RunID is required for run/result. A returned URL is private; do not share it.
arguments
    project (1,1) opendpd.Project
    options.Page (1,1) string {mustBeMember(options.Page, ...
        ["home", "matlink", "datasets", "experiments", "new-experiment", "results", "run", "result"])} = "home"
    options.RunID (1,1) string = ""
    options.OpenBrowser (1,1) logical = true
end
if options.OpenBrowser
    module = bridge();
    service = jsondecode(char(module.encode(project.Backend.studio_info())));
    if ~service.ready
        error('opendpd:StudioNotReady', ...
            'Studio is not ready: %s. For source installs, build the frontend as described in opendpd.help.', ...
            strjoin(string(service.problems), '; '));
    end
end
if strlength(options.RunID) > 0
    url = string(project.Backend.studio_url(char(options.Page), char(options.RunID)));
else
    url = string(project.Backend.studio_url(char(options.Page)));
end
if options.OpenBrowser && web(char(url), '-browser') ~= 0
    error('opendpd:Browser', 'The system browser could not be opened.');
end
if nargout == 0
    clear url; % do not print a private bootstrap URL in the Command Window
end
end
