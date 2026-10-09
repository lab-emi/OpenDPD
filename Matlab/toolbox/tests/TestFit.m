classdef TestFit < matlab.unittest.TestCase
    % opendpd.fit through the process transport: Python is started as a child process, pyenv is not used.
    % Needs OPENDPD_MATLAB_PYTHON (a Python with this OpenDPD checkout installed); OPENDPD_SOURCE may name the checkout.
    properties
        Python
        Source
        X
        Y
        Training = struct('epochs', 1, 'frame_length', 32, 'frame_stride', 32, 'batch_size', 16, 'batch_size_eval', 16)
    end
    methods (TestClassSetup)
        function environment(testCase)
            testCase.Python = string(getenv('OPENDPD_MATLAB_PYTHON'));
            testCase.assertNotEmpty(char(testCase.Python), 'Set OPENDPD_MATLAB_PYTHON to the Python environment with this checkout installed.');
            testCase.Source = string(getenv('OPENDPD_SOURCE'));
            rng(12);
            testCase.X = complex(single(randn(4096, 1)), single(randn(4096, 1))) / 8;
            testCase.Y = testCase.X - single(0.2) * testCase.X .* abs(testCase.X) .^ 2;
        end
    end
    methods (Test)
        function everythingIsCheckedBeforeAnythingStarts(testCase)
            ws = fullfile(string(tempname), "never-created");
            common = {'SampleRate', 80e6, 'Bandwidth', 20e6, 'SegmentSamples', 128};
            x = testCase.X; y = testCase.Y;
            testCase.verifyError(@() opendpd.fit(x, y, common{:}), 'opendpd:Workspace');
            testCase.verifyError(@() opendpd.fit(x, y, Workspace=ws, SampleRate=80e6, Bandwidth=20e6), 'opendpd:SignalMetadata');
            testCase.verifyError(@() opendpd.fit(x, y, Workspace=ws, Bandwidth=20e6, SegmentSamples=128), 'opendpd:SignalMetadata');
            testCase.verifyError(@() opendpd.fit(x, y(1:end-1), common{:}, Workspace=ws), 'opendpd:InvalidIQ');
            testCase.verifyError(@() opendpd.fit(x, [y(1:end-1); NaN], common{:}, Workspace=ws), 'opendpd:InvalidIQ');
            testCase.verifyError(@() opendpd.fit(x, y, common{:}, Workspace=ws, DPDModel="lstm"), 'opendpd:UnsupportedModel');
            testCase.verifyError(@() opendpd.fit(x, y, common{:}, Workspace=ws, PAModel="mp_ls"), 'opendpd:UnsupportedModel');
            testCase.verifyError(@() opendpd.fit(x, y, common{:}, Workspace=ws, PythonExecutable=fullfile(tempdir, 'no-python')), ...
                'opendpd:PythonEnvironment');
            testCase.verifyFalse(isfolder(ws), 'a refused call must not create the workspace');
            try
                opendpd.fit(x, y, Workspace=ws, SampleRate=80e6, Bandwidth=20e6);
            catch cause
                testCase.verifySubstring(cause.message, 'SegmentSamples');
                testCase.verifySubstring(cause.message, 'restarts');
            end
        end

        function fitTrainsAndReturnsModelsThatRunInPlainMatlab(testCase)
            workspace = fullfile(string(tempname), "workspace");
            testCase.addTeardown(@() TestFit.remove(fileparts(workspace)));
            pyenvBefore = char(pyenv().Status);
            output = evalc(['[dpd, pa, report] = opendpd.fit(testCase.X, testCase.Y, Workspace=workspace, SampleRate=80e6, ' ...
                'Bandwidth=20e6, SegmentSamples=128, PAModel="gru", DPDModel="gmp", PAParameters=struct(''hidden_size'', 4), ' ...
                'Training=testCase.Training, Device="cpu", PythonExecutable=testCase.Python, SourceDirectory=testCase.Source);']);
            testCase.verifySubstring(output, 'opendpd.fit: importing the capture');
            testCase.verifySubstring(output, 'exporting the DPD model package');
            testCase.verifyEqual(char(pyenv().Status), pyenvBefore, 'fit must not load Python into MATLAB');
            testCase.verifyClass(dpd, 'opendpd.Model');
            testCase.verifyClass(pa, 'opendpd.Model');
            testCase.verifyEqual(dpd.Manifest.model.key, 'gmp');
            testCase.verifyEqual(dpd.Manifest.run.role, 'dpd');
            testCase.verifyEqual(pa.Manifest.model.key, 'gru');
            testCase.verifyEqual(pa.Manifest.run.role, 'pa');
            testCase.verifyTrue(report.PA.Verify.passed && report.DPD.Verify.passed);
            testCase.verifyEqual(dpd.Manifest.run.run_id, char(report.DPD.RunID));
            testCase.verifyTrue(isfile(report.DPD.Package) && isfile(report.PA.Package));
            testCase.verifyEqual(dpd.Source, report.DPD.Package);
            testCase.verifyTrue(startsWith(report.DPD.Package, string(java.io.File(char(workspace)).getCanonicalPath())) || ...
                startsWith(report.DPD.Package, workspace));
            testCase.verifyTrue(isfile(fullfile(workspace, 'runs', report.DPD.RunID, 'run.json')), 'the run is an ordinary workspace run');
            testCase.verifyFalse(isfolder(report.Job), 'the job folder is removed on success');
            testCase.verifyEqual(report.Dataset.n_samples, numel(testCase.X));
            u = opendpd.apply(dpd, testCase.X(1:300));
            testCase.verifySize(u, [300 1]);
            testCase.verifyClass(u, 'single');
            streamed = dpd(testCase.X(1:300));                          % gmp has a streaming variant
            testCase.verifySize(streamed, [300 1]);
            y = opendpd.apply(pa, u);
            testCase.verifySize(y, [300 1]);
        end

        function aTimeoutCancelsTheRunAndSaysSo(testCase)
            workspace = fullfile(string(tempname), "workspace");
            testCase.addTeardown(@() TestFit.remove(fileparts(workspace)));
            long = testCase.Training;
            long.epochs = 400;
            x = testCase.X; y = testCase.Y; python = testCase.Python; source = testCase.Source;
            call = @() opendpd.fit(x, y, Workspace=workspace, SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128, ...
                PAParameters=struct('hidden_size', 4), Training=long, Device="cpu", Timeout=1, Verbose=false, ...
                PythonExecutable=python, SourceDirectory=source);
            testCase.verifyError(call, 'opendpd:Timeout');
            states = dir(fullfile(workspace, 'runs', '*', 'run.json'));
            testCase.assertNotEmpty(states, 'the PA run should exist');
            for k = 1:numel(states)
                record = jsondecode(fileread(fullfile(states(k).folder, states(k).name)));
                testCase.verifyEqual(record.status, 'cancelled');
            end
            testCase.verifyEmpty(dir(fullfile(workspace, 'exports', '*.zip')));
        end

        function aFailingPythonReportsItsOwnMessage(testCase)
            % A Python without OpenDPD: the module is missing; the error carries what Python said and where the log is.
            testCase.assumeTrue(isunix, 'needs a shell wrapper');
            nothing = fullfile(string(tempname), "workspace");
            testCase.addTeardown(@() TestFit.remove(fileparts(nothing)));
            empty = string(tempname);
            mkdir(empty);
            testCase.addTeardown(@() TestFit.remove(empty));
            wrapper = fullfile(empty, "python-without-opendpd");
            fid = fopen(wrapper, 'w');
            fprintf(fid, '#!/bin/sh\nexec "%s" -I -S "$@"\n', testCase.Python);     % -I: ignore PYTHONPATH and user site; -S: no site-packages, so an installed OpenDPD is not found either
            fclose(fid);
            fileattrib(wrapper, '+x');
            x = testCase.X; y = testCase.Y;
            try
                opendpd.fit(x, y, Workspace=nothing, SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128, PythonExecutable=wrapper, Verbose=false);
                testCase.verifyFail('fit should have failed');
            catch cause
                testCase.verifyEqual(cause.identifier, 'opendpd:FitFailed');
                testCase.verifySubstring(cause.message, 'fit stopped');
                testCase.verifySubstring(cause.message, 'opendpd');
            end
        end
    end
    methods (Static, Access = private)
        function remove(folder)
            if isfolder(folder)
                rmdir(folder, 's');
            end
        end
    end
end
