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
% Tests that need MATLAB Coder, Simulink or a toolbox skip themselves where it is missing, and assertSuccess accepts that. A CI job
% that installs those products sets OPENDPD_REQUIRE_ALL_TESTS=1, so that a product which failed to install is an error and not a
% silent loss of coverage. OPENDPD_ALLOW_SKIPPED is a regular expression (case-insensitive) for the names of the tests that may skip
% all the same: the ones whose product that CI cannot license.
if strcmp(getenv('OPENDPD_REQUIRE_ALL_TESTS'), '1')
    skipped = results([results.Incomplete]);
    allowed = getenv('OPENDPD_ALLOW_SKIPPED');
    if ~isempty(skipped) && ~isempty(allowed)
        skipped = skipped(cellfun(@isempty, regexpi({skipped.Name}, allowed, 'once')));
    end
    if ~isempty(skipped)
        error('opendpd:build:TestsSkipped', '%d tests were skipped, and OPENDPD_REQUIRE_ALL_TESTS=1 allows none of them to be:\n%s', ...
            numel(skipped), strjoin(string({skipped.Name}), newline));
    end
end
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
opts.ToolboxVersion = '2.4.0';      % follows the OpenDPD release it ships with
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
opts.OutputFile = fullfile(out, ['OpenDPD-' char(opts.ToolboxVersion) '.mltbx']);
matlab.addons.toolbox.packageToolbox(opts);
end
