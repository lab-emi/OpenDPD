classdef TestTransport < matlab.unittest.TestCase
    % The process transport: Python as a child process of MATLAB (no pyenv). Needs a Python executable with NumPy in
    % OPENDPD_MATLAB_PYTHON; it does not need OpenDPD for these tests.
    properties
        Python
        Folder
    end
    methods (TestClassSetup)
        function environment(testCase)
            testCase.Python = string(getenv('OPENDPD_MATLAB_PYTHON'));
            testCase.assertNotEmpty(char(testCase.Python), 'Set OPENDPD_MATLAB_PYTHON to a Python with NumPy.');
        end
    end
    methods (TestMethodSetup)
        function folder(testCase)
            testCase.Folder = string(testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
        end
    end
    methods (Test)
        function npyWrittenByMatlabIsReadByNumPyAndByOurReader(testCase)
            stream = RandStream('twister', Seed=3);
            cases = {single(randn(stream, 7, 2)), randn(stream, 5, 2), randn(stream, 6, 1), single(randn(stream, 2, 3, 4)), ...
                randn(stream, 1, 5), single(7), randn(stream, 4096, 2)};
            script = ['import numpy as np, json, sys; a = np.load(sys.argv[1], allow_pickle=False); ' ...
                'print(json.dumps({"dtype": str(a.dtype), "shape": list(a.shape), "flat": a.reshape(-1).astype("float64").tolist()}))'];
            for k = 1:numel(cases)
                a = cases{k};
                file = fullfile(testCase.Folder, sprintf('a%d.npy', k));
                opendpd.internal.writeNpy(file, a);
                back = opendpd.internal.readNpy(PackageTools.readBytes(file), 'a.npy');
                expected = a;
                if isvector(a) && ~isscalar(a), expected = a(:); end
                testCase.verifyEqual(back, expected, sprintf('case %d: our reader', k));
                testCase.verifyEqual(PackageTools.readBytes(file), PackageTools.npy(a), sprintf('case %d: same bytes as the test writer', k));
                outcome = opendpd.internal.runPython(testCase.Python, ["-c", script, file], Log=fullfile(testCase.Folder, 'log.txt'));
                testCase.assertEqual(outcome.ExitCode, 0, outcome.Log);
                seen = jsondecode(strtrim(outcome.Log));
                testCase.verifyEqual(seen.dtype, ternary(isa(a, 'single'), 'float32', 'float64'));
                shape = size(a);
                if isvector(a) && ~isscalar(a), shape = numel(a); elseif isscalar(a), shape = []; end
                testCase.verifyEqual(double(seen.shape(:)).', double(shape(:)).', sprintf('case %d: NumPy shape', k));
                % NumPy reads C order: the flat list is a's elements with the last index fastest
                flat = permute(a, ndims(a):-1:1);
                testCase.verifyEqual(seen.flat(:), double(flat(:)), sprintf('case %d: NumPy values', k), AbsTol=0);
            end
        end

        function writeNpyRefusesWhatItCannotRepresent(testCase)
            file = fullfile(testCase.Folder, 'bad.npy');
            testCase.verifyError(@() opendpd.internal.writeNpy(file, [1 NaN]), 'MATLAB:validators:mustBeFinite');
            testCase.verifyError(@() opendpd.internal.writeNpy(file, complex(1, 2)), 'MATLAB:validators:mustBeReal');
            testCase.verifyError(@() opendpd.internal.writeNpy(file, int8([1 2])), 'opendpd:InvalidIQ');
        end

        function runPythonReportsExitCodeAndLog(testCase)
            log = fullfile(testCase.Folder, 'log.txt');
            ok = opendpd.internal.runPython(testCase.Python, ["-c", "print('hello'); import sys; print('warn', file=sys.stderr)"], Log=log);
            testCase.verifyEqual(ok.ExitCode, 0);
            testCase.verifySubstring(ok.Log, 'hello');
            testCase.verifySubstring(ok.Log, 'warn');                       % stderr is in the same log
            testCase.verifyFalse(ok.TimedOut);
            bad = opendpd.internal.runPython(testCase.Python, ["-c", "raise SystemExit(3)"], Log=log);
            testCase.verifyEqual(bad.ExitCode, 3);
            missing = fullfile(testCase.Folder, 'no-such-python');
            testCase.verifyError(@() opendpd.internal.runPython(missing, ["-c", "pass"], Log=log), 'opendpd:PythonEnvironment');
            testCase.verifyError(@() opendpd.internal.runPython(testCase.Python, ["-c", "pass"]), 'opendpd:Transport');
        end

        function runPythonPassesEnvironmentAndWorkingDirectory(testCase)
            script = "import os, json; print(json.dumps({'a': os.environ.get('OPENDPD_TEST_VALUE'), 'cwd': os.getcwd()}))";
            outcome = opendpd.internal.runPython(testCase.Python, ["-c", script], Log=fullfile(testCase.Folder, 'log.txt'), ...
                Environment=struct('OPENDPD_TEST_VALUE', 'x y'), WorkingDirectory=testCase.Folder);
            seen = jsondecode(strtrim(outcome.Log));
            testCase.verifyEqual(seen.a, 'x y');
            testCase.verifyEqual(string(java.io.File(seen.cwd).getCanonicalPath()), string(java.io.File(char(testCase.Folder)).getCanonicalPath()));
        end

        function aTimeoutAsksTheProgramToStopThenKillsIt(testCase)
            cancel = fullfile(testCase.Folder, 'cancel');
            polite = string(sprintf(['import os, sys, time\nt0 = time.time()\n' ...
                'while not os.path.exists(sys.argv[1]) and time.time() - t0 < 60: time.sleep(0.05)\nraise SystemExit(7)']));
            started = tic;
            outcome = opendpd.internal.runPython(testCase.Python, ["-c", polite, cancel], Log=fullfile(testCase.Folder, 'log.txt'), ...
                Timeout=1, CancelFile=cancel, Grace=20, PollInterval=0.1);
            testCase.verifyTrue(outcome.TimedOut);
            testCase.verifyEqual(outcome.ExitCode, 7, 'the program saw the cancel file and stopped itself');
            testCase.verifyLessThan(toc(started), 15);
            testCase.verifyTrue(isfile(cancel), 'the cancel file is how the program is asked to stop');
            delete(cancel);
            stubborn = string(sprintf('import time\nt0 = time.time()\nwhile time.time() - t0 < 60: time.sleep(0.05)'));
            started = tic;
            outcome = opendpd.internal.runPython(testCase.Python, ["-c", stubborn], Log=fullfile(testCase.Folder, 'log2.txt'), ...
                Timeout=1, CancelFile=cancel, Grace=1, PollInterval=0.1);
            testCase.verifyTrue(outcome.TimedOut);
            testCase.verifyNotEqual(outcome.ExitCode, 0, 'a program that ignores the request is killed');
            testCase.verifyLessThan(toc(started), 15);
        end

        function anInterruptedWaitStopsTheChild(testCase)
            % Ctrl+C in MATLAB raises an error inside the polling loop; the cleanup must stop the child, not orphan it.
            cancel = fullfile(testCase.Folder, 'cancel');
            marker = fullfile(testCase.Folder, 'stopped');
            polite = string(sprintf(['import os, sys, time\nt0 = time.time()\n' ...
                'while not os.path.exists(sys.argv[1]) and time.time() - t0 < 60: time.sleep(0.05)\n' ...
                'open(sys.argv[2], "w").close() if os.path.exists(sys.argv[1]) else None']));      % never outlives the test by more than a minute
            interrupt = @() error('test:interrupt', 'simulated Ctrl+C');
            testCase.verifyError(@() opendpd.internal.runPython(testCase.Python, ["-c", polite, cancel, marker], ...
                Log=fullfile(testCase.Folder, 'log.txt'), CancelFile=cancel, Grace=20, PollInterval=0.1, OnPoll=interrupt), 'test:interrupt');
            testCase.verifyTrue(isfile(marker), 'the child was asked to stop and did');
        end

        function matlabLibraryFoldersAreRemovedFromTheChildPath(testCase)
            root = "/opt/matlab/R2026a";
            value = "/opt/matlab/R2026a/sys/os/glnxa64:/usr/local/cuda/lib64:/opt/matlab/R2026a:/opt/matlab/R2026a2/lib:/home/u/lib";
            kept = opendpd.internal.withoutMatlabEntries(value, root, ":");
            testCase.verifyEqual(kept, '/usr/local/cuda/lib64:/opt/matlab/R2026a2/lib:/home/u/lib');
            testCase.verifyEqual(opendpd.internal.withoutMatlabEntries("/opt/matlab/R2026a/bin", root, ":"), '');
            testCase.verifyEqual(opendpd.internal.withoutMatlabEntries("", root, ":"), '');
            testCase.verifyEqual(opendpd.internal.withoutMatlabEntries("C:\M\bin;C:\Python;C:\M", "C:\M", ";"), 'C:\Python');
        end

        function thePythonExecutableIsResolvedInAFixedOrder(testCase)
            [fake, other] = fakePythons(testCase.Folder);
            checkResolutionOrder(testCase, fake, other);
        end

        function theProgressPrinterShowsProgressOnlyWhenAskedAndAlwaysWarns(testCase)
            file = fullfile(testCase.Folder, 'progress.jsonl');
            lines = {'{"stage":"import"}', ...
                'this line is not JSON', ...
                '{"stage":"train_pa","run_id":"r1","status":"running","epoch":null,"total_epochs":null}', ...
                '{"stage":"train_pa","run_id":"r1","status":"running","epoch":2,"total_epochs":2}', ...
                '{"stage":"export","role":"pa","run_id":"r1"}', ...
                '{"stage":"close","warning":"shutdown_timeout: Service is still stopping"}'};
            fid = fopen(file, 'w'); fprintf(fid, '%s\n', strjoin(string(lines), newline)); fclose(fid);   % one line is garbage on purpose
            previous = warning('off', 'opendpd:fit:ServiceNotStopped');         % lastwarn still records a disabled warning
            testCase.addTeardown(@() warning(previous));
            loud = opendpd.internal.ProgressPrinter(file, true, "gru", "gru");
            lastwarn('');
            text = evalc('loud.poll()');
            testCase.verifySubstring(text, 'importing the capture');
            testCase.verifySubstring(text, 'PA (gru) epoch 2 of 2');
            testCase.verifySubstring(text, 'exporting the PA model package');
            [message, id] = lastwarn();
            testCase.verifyEqual(id, 'opendpd:fit:ServiceNotStopped');
            testCase.verifySubstring(message, 'Service is still stopping');
            quiet = opendpd.internal.ProgressPrinter(file, false, "gru", "gru");
            lastwarn('');
            text = evalc('quiet.poll()');
            testCase.verifyEmpty(strtrim(text), 'Verbose=false prints no progress lines');
            [~, id] = lastwarn();
            testCase.verifyEqual(id, 'opendpd:fit:ServiceNotStopped', 'a service that did not stop is reported even when quiet');
            % a line that is still being written is left for the next poll, and nothing is shown twice
            fid = fopen(file, 'a'); fprintf(fid, '{"stage":"export","role":"dpd","run_id":"r'); fclose(fid);
            text = evalc('quiet.poll(); loud.poll()');
            testCase.verifyEmpty(strtrim(text));
            fid = fopen(file, 'a'); fprintf(fid, '2"}\n'); fclose(fid);
            text = evalc('loud.poll()');
            testCase.verifySubstring(text, 'exporting the DPD model package');
        end

        function theResolutionTestLeavesTheUsersSettingsAsItFoundThem(testCase)
            % Guards the guard: changing OPENDPD_PYTHON and the remembered Python to test their order must not leak out
            % of the test. Both may be absent in a fresh MATLAB profile, and both may be set in a user's.
            [fake, other] = fakePythons(testCase.Folder);
            marker = "sentinel-" + string(randi(1e6));
            originalEnvironment = getenv('OPENDPD_PYTHON');
            hadPref = ispref('OpenDPDToolbox', 'Python');
            originalPref = '';
            if hadPref
                originalPref = getpref('OpenDPDToolbox', 'Python');
            end
            testCase.addTeardown(@() restoreUserSettings(originalEnvironment, hadPref, originalPref));
            setenv('OPENDPD_PYTHON', char(marker));
            setpref('OpenDPDToolbox', 'Python', char(marker));
            checkResolutionOrder(testCase, fake, other);
            testCase.verifyEqual(string(getenv('OPENDPD_PYTHON')), marker, 'OPENDPD_PYTHON is put back');
            testCase.verifyTrue(ispref('OpenDPDToolbox', 'Python'), 'the remembered Python still exists');
            testCase.verifyEqual(string(getpref('OpenDPDToolbox', 'Python')), marker, 'the remembered Python is put back');
        end
    end
end

function value = ternary(condition, a, b)
if condition
    value = a;
else
    value = b;
end
end

function [fake, other] = fakePythons(folder)
% Two files that exist (the resolver checks existence, not that they run).
fake = fullfile(folder, 'python-from-environment');
fid = fopen(fake, 'w'); fclose(fid);
other = fullfile(folder, 'python-from-pref');
fid = fopen(other, 'w'); fclose(fid);
end

function checkResolutionOrder(testCase, fake, other)
% Changes OPENDPD_PYTHON and the remembered Python, checks the resolution order, and puts both back when it returns -
% also when it errors, because the restore is an onCleanup that captures the values BEFORE they are changed.
originalEnvironment = getenv('OPENDPD_PYTHON');
hadPref = ispref('OpenDPDToolbox', 'Python');
originalPref = '';
if hadPref
    originalPref = getpref('OpenDPDToolbox', 'Python');
end
restore = onCleanup(@() restoreUserSettings(originalEnvironment, hadPref, originalPref)); %#ok<NASGU>
setenv('OPENDPD_PYTHON', char(fake));
setpref('OpenDPDToolbox', 'Python', char(other));
testCase.verifyEqual(opendpd.internal.pythonExecutable(testCase.Python), testCase.Python);       % 1 explicit
testCase.verifyEqual(opendpd.internal.pythonExecutable(), string(fake));                          % 2 environment
setenv('OPENDPD_PYTHON', '');
testCase.verifyEqual(opendpd.internal.pythonExecutable(), string(other));                         % 3 remembered
testCase.verifyError(@() opendpd.internal.pythonExecutable(fullfile(testCase.Folder, 'missing')), 'opendpd:PythonEnvironment');
end

function restoreUserSettings(environmentValue, hadPref, prefValue)
setenv('OPENDPD_PYTHON', environmentValue);
if hadPref
    setpref('OpenDPDToolbox', 'Python', prefValue);
elseif ispref('OpenDPDToolbox', 'Python')
    rmpref('OpenDPDToolbox', 'Python');
end
end
