function [dpd, pa, report] = fit(x, y, options)
%FIT Train a PA model and a DPD from paired I/Q in one call and get both back as MATLAB objects.
%   [dpd, pa, report] = opendpd.fit(x, y, Workspace="my-workspace", SampleRate=fs, Bandwidth=bw, SegmentSamples=2048)
% x is the PA input and y the PA output (complex vectors of the same length, nothing is normalised). Training runs in
% Python/PyTorch from the usual OpenDPD workspace, so every step is an ordinary run you can open in Studio later; the
% results come back as opendpd.Model objects that run in plain MATLAB (opendpd.apply(dpd, xNew), streaming for gru
% and gmp). Python is started as a separate process: MATLAB's pyenv is not used or loaded, so the Python version need
% not match the MATLAB release. Pass PythonExecutable, or set OPENDPD_PYTHON, or run opendpd.studio/opendpd.setup once.
%
% Required: Workspace (a folder; it is created and nothing is stored anywhere else), SampleRate and Bandwidth in Hz, and
% SegmentSamples, which has no default because spectral metrics use it as their Welch segment and evaluation restarts a
% model's state at the same interval (Studio's generated signals use 512-4096).
% Models: DPDModel and PAModel (default "gru") must be exportable (gru, tres_gru, gmp, mp_ls, gmp_ls), and PAModel must
% be gradient-trained (gru, tres_gru, gmp): the DPD is trained through it. Both are checked before anything trains.
% Training, DPDParameters, PAParameters, Device, DeviceIndex and NumThreads are as for opendpd.trainPA/trainDPD.
% Timeout (seconds, default none) cancels the running job. Ctrl+C cancels it too and returns after the job has stopped.
% report has Workspace, Dataset, PA and DPD (RunID, Result, Package file, Verify), Seconds, Python and Job (the folder
% with the progress and log files; it is removed on success).
arguments
    x {mustBeNumeric}
    y {mustBeNumeric}
    options.Workspace (1,1) string
    options.SampleRate (1,1) double {mustBePositive, mustBeFinite}
    options.Bandwidth (1,1) double {mustBePositive, mustBeFinite}
    options.SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)}
    options.DPDModel (1,1) string = "gru"
    options.PAModel (1,1) string = "gru"
    options.DPDParameters (1,1) struct = struct()
    options.PAParameters (1,1) struct = struct()
    options.Training (1,1) struct = struct()
    options.Device (1,1) string {mustBeMember(options.Device, ["auto", "cpu", "cuda", "mps"])} = "auto"
    options.DeviceIndex (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.NumThreads (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.Profile (1,1) string = "opendpd-spectral-v2"
    options.Name (1,1) string = ""
    options.Subchannels (1,1) double {mustBeInteger, mustBePositive} = 1
    options.GuardSamples (1,1) double {mustBeInteger, mustBeNonnegative} = 256
    options.Origin (1,1) string {mustBeMember(options.Origin, ["unknown", "measured", "synthetic"])} = "unknown"
    options.AmplitudeUnits (1,1) string {mustBeMember(options.AmplitudeUnits, ["unknown", "normalized", "volts"])} = "unknown"
    options.OutputFolder (1,1) string = ""
    options.Timeout (1,1) double {mustBePositive} = Inf
    options.PythonExecutable (1,1) string = ""
    options.SourceDirectory (1,1) string = ""
    options.Verbose (1,1) logical = true
end
if ~isfield(options, 'Workspace') || strlength(options.Workspace) == 0
    error('opendpd:Workspace', ['Supply Workspace: the folder for the datasets, runs and model packages (it is created, and ' ...
        'you can open it in Studio later). There is no default so that results never land somewhere you did not choose.']);
end
importOptions(options);                        % SampleRate, Bandwidth and SegmentSamples must be given, with the usual explanation
iqX = opendpd.internal.iqMatrix(x, 'x');
iqY = opendpd.internal.iqMatrix(y, 'y');
if size(iqX, 1) ~= size(iqY, 1)
    error('opendpd:InvalidIQ', 'x and y must have the same number of samples (%d and %d).', size(iqX, 1), size(iqY, 1));
end
exportable = ["gru", "tres_gru", "gmp", "mp_ls", "gmp_ls"];
for model = [options.DPDModel, options.PAModel]
    if ~any(model == exportable)
        error('opendpd:UnsupportedModel', '"%s" cannot be exported; fit supports %s.', model, strjoin(exportable, ', '));
    end
end
if ~any(options.PAModel == ["gru", "tres_gru", "gmp"])
    error('opendpd:UnsupportedModel', ['PAModel "%s" is a least-squares baseline, not a DPD surrogate: the DPD is trained ' ...
        'through the PA model, so PAModel must be gru, tres_gru or gmp.'], options.PAModel);
end

executable = opendpd.internal.pythonExecutable(options.PythonExecutable);
source = options.SourceDirectory;
if strlength(source) == 0
    source = string(getpref('OpenDPDToolbox', 'Source', ''));
end
workspace = absolutePath(options.Workspace);
job = string(tempname);
mkdir(job);
opendpd.internal.writeNpy(fullfile(job, 'x.npy'), iqX);
opendpd.internal.writeNpy(fullfile(job, 'y.npy'), iqY);
request = struct('workspace', workspace, 'sample_rate_hz', options.SampleRate, 'bandwidth_hz', options.Bandwidth, ...
    'nperseg', options.SegmentSamples, 'n_sub_ch', options.Subchannels, 'guard_samples', options.GuardSamples, ...
    'origin', options.Origin, 'amplitude_units', options.AmplitudeUnits, 'dpd_model', options.DPDModel, ...
    'pa_model', options.PAModel, 'dpd_parameters', options.DPDParameters, 'pa_parameters', options.PAParameters, ...
    'training', options.Training, 'device', options.Device, 'device_index', options.DeviceIndex, ...
    'profile', options.Profile, 'poll_interval', 0.5);
if options.NumThreads > 0, request.num_threads = options.NumThreads; end
if strlength(options.Name) > 0, request.dataset_id = options.Name; end
if strlength(options.OutputFolder) > 0, request.output_folder = absolutePath(options.OutputFolder); end
if isfinite(options.Timeout), request.timeout = options.Timeout; end
fid = fopen(fullfile(job, 'request.json'), 'w');
fwrite(fid, jsonencode(request), 'char');
fclose(fid);

environment = struct('PYTHONUNBUFFERED', '1', 'PYTHONIOENCODING', 'utf-8', 'MPLBACKEND', 'Agg', 'TQDM_DISABLE', '1');
if strlength(source) > 0
    environment.PYTHONPATH = char(strjoin(nonEmpty([source, string(getenv('PYTHONPATH'))]), pathsep));
end
progress = opendpd.internal.ProgressPrinter(fullfile(job, 'progress.jsonl'), options.Verbose, options.DPDModel, options.PAModel);
started = tic;
outcome = opendpd.internal.runPython(executable, ["-m", "opendpd.sdk._fit", "--job", job], ...
    Log=fullfile(job, 'log.txt'), Environment=environment, Timeout=options.Timeout + 120, ...
    CancelFile=fullfile(job, 'cancel'), OnPoll=@progress.poll);
failure = fullfile(job, 'error.json');
if outcome.ExitCode ~= 0 || ~isfile(fullfile(job, 'result.json'))
    detail = tail(outcome.Log);
    code = 'opendpd:FitFailed';
    if isfile(failure)
        info = jsondecode(fileread(failure));
        detail = string(info.message);
        code = fitErrorId(info.code);
    elseif outcome.TimedOut
        code = 'opendpd:Timeout';
    end
    error(code, 'fit stopped: %s\nThe workspace %s keeps whatever ran. Progress and log: %s', detail, workspace, job);
end
result = jsondecode(fileread(fullfile(job, 'result.json')));
cleanup = onCleanup(@() rmdir(job, 's'));
dpd = opendpd.load(result.dpd.package.path);
pa = opendpd.load(result.pa.package.path);
verification = struct('PA', opendpd.verify(pa), 'DPD', opendpd.verify(dpd));
for role = ["PA", "DPD"]
    if ~verification.(role).passed
        error('opendpd:Verification', ['The exported %s model does not reproduce OpenDPD''s outputs on this MATLAB release ' ...
            '(max error %g, limit %g). Do not use it; the package is %s.'], role, verification.(role).offline_max_abs_error, ...
            verification.(role).tolerance_abs, result.(lower(role)).package.path);
    end
end
report = struct('Workspace', string(result.workspace), 'Dataset', result.dataset, 'Seconds', toc(started), ...
    'Python', executable, 'Job', job, ...
    'PA', struct('RunID', string(result.pa.run_id), 'Result', result.pa.result, 'Package', string(result.pa.package.path), ...
    'Verify', verification.PA), ...
    'DPD', struct('RunID', string(result.dpd.run_id), 'Result', result.dpd.result, 'Package', string(result.dpd.package.path), ...
    'Verify', verification.DPD));
end

function path = absolutePath(path)
path = string(java.io.File(char(path)).getAbsoluteFile().getPath());
end

function values = nonEmpty(values)
values = values(strlength(values) > 0);
end

function text = tail(log)
text = string(log);
if strlength(text) > 1500
    text = "..." + extractAfter(text, strlength(text) - 1500);
end
text = strtrim(text);
if strlength(text) == 0
    text = "the Python process printed nothing";
end
end

function id = fitErrorId(code)
switch string(code)
    case "cancelled", id = 'opendpd:Cancelled';
    case "timeout", id = 'opendpd:Timeout';
    otherwise, id = 'opendpd:FitFailed';
end
end
