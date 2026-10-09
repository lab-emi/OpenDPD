function outcome = runPython(executable, commandArguments, options)
%RUNPYTHON Start a Python program as a child process of MATLAB and watch it, without loading Python into MATLAB.
% The process transport: no pyenv, no in-process interpreter, so the Python version need not match the MATLAB release
% and a crash in Python cannot take MATLAB down. Output goes to LOG (never a pipe MATLAB would have to drain).
%   OnPoll         function handle called every PollInterval seconds while the process runs
%   CancelFile     a file the program watches; it is created when MATLAB is interrupted (Ctrl+C) or Timeout passes
%   Grace          seconds the program gets to stop after that before it is killed
% Returns a struct: ExitCode, TimedOut, Log (text), Seconds.
arguments
    executable (1,1) string
    commandArguments (1,:) string
    options.Log (1,1) string = ""
    options.WorkingDirectory (1,1) string = ""
    options.Environment (1,1) struct = struct()
    options.Timeout (1,1) double {mustBePositive} = Inf
    options.CancelFile (1,1) string = ""
    options.Grace (1,1) double {mustBePositive} = 60
    options.PollInterval (1,1) double {mustBePositive} = 0.5
    options.OnPoll = []
end
command = java.util.ArrayList();
command.add(char(executable));
if strlength(options.Log) == 0
    error('opendpd:Transport', 'runPython needs a Log file.');
end
for k = 1:numel(commandArguments)
    command.add(char(commandArguments(k)));
end
builder = java.lang.ProcessBuilder(command);
environment = builder.environment();
for name = ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "PATH"]
    current = environment.get(char(name));
    if ~isempty(current) && (name ~= "PATH" || ispc)
        cleaned = opendpd.internal.withoutMatlabEntries(string(current), string(matlabroot));
        if isempty(cleaned)
            environment.remove(char(name));
        else
            environment.put(char(name), cleaned);
        end
    end
end
for name = string(fieldnames(options.Environment)).'
    environment.put(char(name), char(string(options.Environment.(name))));
end
if strlength(options.WorkingDirectory) > 0
    builder.directory(java.io.File(char(options.WorkingDirectory)));
end
builder.redirectErrorStream(true);
builder.redirectOutput(java.io.File(char(options.Log)));
started = tic;
try
    process = builder.start();
catch cause
    error('opendpd:PythonEnvironment', 'Cannot start %s: %s', executable, cause.message);
end
stop = onCleanup(@() stopIfRunning(process, options.CancelFile, options.Grace));     % Ctrl+C, errors in OnPoll, ...
timedOut = false;
while process.isAlive()
    pause(options.PollInterval);
    if ~isempty(options.OnPoll)
        options.OnPoll();
    end
    if ~timedOut && toc(started) > options.Timeout
        timedOut = true;
        requestStop(options.CancelFile);
        graceEnds = toc(started) + options.Grace;
    end
    if timedOut && toc(started) > graceEnds
        process.destroyForcibly();
        break
    end
end
process.waitFor();
if ~isempty(options.OnPoll)
    options.OnPoll();
end
outcome = struct('ExitCode', double(process.exitValue()), 'TimedOut', timedOut, ...
    'Log', readLog(options.Log), 'Seconds', toc(started));
end

function requestStop(cancelFile)
if strlength(cancelFile) > 0
    fid = fopen(cancelFile, 'w');
    if fid >= 0
        fclose(fid);
    end
end
end

function stopIfRunning(process, cancelFile, grace)
% Runs when runPython is left: normally the process is gone; after Ctrl+C it is asked to stop, then killed.
if ~process.isAlive()
    return
end
requestStop(cancelFile);
fprintf('Stopping the Python process (up to %g s)...\n', grace);
waited = tic;
while process.isAlive() && toc(waited) < grace
    pause(0.2);
end
if process.isAlive()
    process.destroyForcibly();
end
end

function text = readLog(path)
if isfile(path)
    text = fileread(path);
else
    text = '';
end
end
