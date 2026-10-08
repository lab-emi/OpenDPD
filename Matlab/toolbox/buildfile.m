function plan = buildfile
%BUILDFILE Run the MATLAB tests and package a development toolbox.
% From this folder: buildtool test or buildtool package.
plan = buildplan(localfunctions);
plan.DefaultTasks = "test";
plan("package").Dependencies = "test";
end

function testTask(context)
% Run real MATLAB/Python integration tests.
root = context.Plan.RootFolder;
addpath(root);
results = runtests(fullfile(root, 'tests'), IncludeSubfolders=true);
assertSuccess(results);
end

function packageTask(context)
% Build a source-only .mltbx after the test task succeeds.
root = context.Plan.RootFolder;
out = fullfile(root, 'dist');
if ~isfolder(out)
    mkdir(out);
end
opts = matlab.addons.toolbox.ToolboxOptions(root, 'a58c7792-3536-49ae-9663-04fb04bcc350');
opts.ToolboxName = 'OpenDPD Toolbox for MATLAB';
opts.ToolboxVersion = '0.4.0';
opts.AuthorName = 'OpenDPD contributors';
opts.AuthorCompany = 'Lab of Efficient Machine Intelligence, TU Delft';
opts.Summary = 'OpenDPD Studio with MATLINK: MATLAB signals and reports in one web GUI.';
opts.MinimumMatlabRelease = 'R2024b';
opts.SupportedPlatforms.MatlabOnline = false;
opts.ToolboxMatlabPath = {char(root)};
opts.ToolboxFiles = {char(fullfile(root, '+opendpd')), char(fullfile(root, 'examples')), ...
    char(fullfile(root, 'tests')), char(fullfile(root, 'README.md')), ...
    char(fullfile(root, 'LICENSE')), char(fullfile(root, 'Contents.m')), ...
    char(fullfile(root, 'buildfile.m')), char(fullfile(root, 'OpenDPDStudio.m')), ...
    char(fullfile(root, 'resources')), char(fullfile(root, 'docs'))};
opts.AppGalleryFiles = {char(fullfile(root, 'OpenDPDStudio.m'))};
opts.ToolboxGettingStartedGuide = fullfile(root, 'examples', 'opendpdQuickstart.m');
if isprop(opts, 'Readme')
    opts.Readme = fullfile(root, 'README.md');
elseif isprop(opts, 'Description')
    opts.Description = fileread(fullfile(root, 'README.md'));
end
opts.OutputFile = fullfile(out, 'OpenDPD-0.4.0.mltbx');
matlab.addons.toolbox.packageToolbox(opts);
end
