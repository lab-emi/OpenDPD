classdef TestToolbox < matlab.unittest.TestCase
    % Real MATLAB/Python integration. Set OPENDPD_MATLAB_PYTHON before running.
    properties
        Project
    end
    methods (TestClassSetup)
        function environment(testCase)
            executable = string(getenv('OPENDPD_MATLAB_PYTHON'));
            testCase.assertNotEmpty(char(executable), ...
                'Set OPENDPD_MATLAB_PYTHON to the Python environment with this checkout installed.');
            info = opendpd.setup(PythonExecutable=executable);
            testCase.assertTrue(info.ok);
            testCase.Project = opendpd.openProject(string(tempname) + " workspace");
            testCase.addTeardown(@() opendpd.closeProject(testCase.Project, StopService=true));
        end
    end
    methods (Test)
        function diagnostics(testCase)
            info = opendpd.doctor();
            testCase.verifyEqual(info.api_version, 1);
            testCase.verifyTrue(info.ok);
        end

        function sourceDirectory(testCase)
            module = py.importlib.import_module('opendpd');
            root = fileparts(fileparts(char(py.getattr(module, '__file__'))));
            info = opendpd.setup(SourceDirectory=string(root));
            testCase.verifyTrue(info.ok);
        end

        function invalidSourceDirectory(testCase)
            testCase.verifyError(@() opendpd.setup(SourceDirectory=string(tempdir)), ...
                'opendpd:SourceDirectory');
        end

        function invalidArrays(testCase)
            p = testCase.Project;
            testCase.verifyError(@() opendpd.importIQ(p, [1, NaN], [1, 2], ...
                SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128), 'opendpd:InvalidIQ');
        end

        function segmentLengthHasNoDefault(testCase)
            p = testCase.Project;
            testCase.verifyError(@() opendpd.importIQ(p, complex(ones(512, 1)), complex(ones(512, 1)), ...
                SampleRate=80e6, Bandwidth=20e6), 'opendpd:SignalMetadata');
            try
                opendpd.importIQ(p, complex(ones(512, 1)), complex(ones(512, 1)), SampleRate=80e6, Bandwidth=20e6);
            catch cause
                testCase.verifySubstring(cause.message, 'SegmentSamples');
                testCase.verifySubstring(cause.message, 'restarts');
            end
        end

        function aLabRecordTravelsWithTheCaptureItProduced(testCase)
            % A measured pair is imported with the compact session record as its source: the dataset then keeps who armed
            % the session, its limits and the hash of what was played and captured, within the notes' 2000 characters.
            p = testCase.Project;
            rng(5);
            u = complex(single(randn(2048, 1)), single(randn(2048, 1))) / 16;
            lab = opendpd.lab.Session(Instrument=opendpd.lab.MockInstrument(), Operator=string(repmat('N', 1, 400)), MaxPeak=0.9);
            arm(lab);
            y = measure(lab, u, SampleRate=80e6);
            disarm(lab);
            record = lab.record(Compact=true);
            ds = opendpd.importIQ(p, u, y, SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128, Origin="measured", ...
                Name="lab-record-test", Source=record);
            notes = jsondecode(ds.notes);
            testCase.verifyEqual(ds.origin, 'measured');
            testCase.verifyLessThan(strlength(ds.notes), 2000);
            testCase.verifyEqual(notes.source.record_sha256, record.record_sha256);
            testCase.verifyEqual(notes.source.final_state, 'disarmed');
            testCase.verifyTrue(notes.source.mock, 'a dry run says so in the dataset');
            testCase.verifyEqual(notes.source.last.captured_sha256, opendpd.lab.iqHash(y));
            testCase.verifyEqual(notes.source.last.played_sha256, opendpd.lab.iqHash(u));
            saved = lab.saveRecord(fullfile(string(testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder), "s.json"));
            testCase.verifyEqual(notes.source.record_sha256, saved.sha256, 'the notes name the complete record by its hash');
        end

        function trainApplyAndReconnect(testCase)
            p = testCase.Project;
            rng(12);
            x = complex(single(randn(4096,1)), single(randn(4096,1))) / 8;
            y = x - single(0.2) * x .* abs(x).^2;
            % A row input and column output must preserve identical sample order.
            ds = opendpd.importIQ(p, x.', y, SampleRate=80e6, Bandwidth=20e6, ...
                Origin="synthetic", Name="matlab-test", SegmentSamples=128);
            testCase.verifyEqual(ds.n_samples, numel(x));
            training = struct('epochs', 1, 'frame_length', 32, 'frame_stride', 32, ...
                'batch_size', 16, 'batch_size_eval', 16);
            parameters = struct('hidden_size', 4);
            pa = opendpd.wait(opendpd.trainPA(p, ds, ModelParameters=parameters, Training=training, Device="cpu"));
            dpd = opendpd.wait(opendpd.trainDPD(p, ds, PA=pa, ModelParameters=parameters, Training=training, Device="cpu"));
            xTest = x(1:257);
            [u, info] = opendpd.apply(dpd, xTest.', Execution="offline_segmented");
            reference = dpd.Backend.apply(py.numpy.asarray([real(xTest), imag(xTest)]), pyargs('execution', 'offline_segmented'));
            referenceIQ = single(reference{1});
            testCase.verifyEqual(u, complex(referenceIQ(:,1), referenceIQ(:,2)), AbsTol=single(1e-6));
            testCase.verifySize(u, [257, 1]);
            testCase.verifyClass(u, 'single');
            testCase.verifyEqual(info.execution, 'offline_segmented');
            testCase.verifyEqual(info.output_role, 'predistorted_pa_input');
            testCase.verifyEqual(info.model, 'gru');
            % Streaming carries one state across the waveform, so it differs from the segmented scoring semantics.
            [s, streamInfo] = opendpd.apply(dpd, xTest.', Execution="streaming", ChunkSamples=50);
            testCase.verifySize(s, [257, 1]);
            testCase.verifyEqual(streamInfo.execution, 'streaming_stateful');
            testCase.verifyEqual(streamInfo.streaming.chunk_samples, 50);
            testCase.verifyTrue(streamInfo.streaming.consistency.within_tolerance);
            testCase.verifyGreaterThan(max(abs(s(129:end) - u(129:end))), 1e-6);
            % The default, "auto", is that streaming execution for a gru (it has a streaming variant), not the scored form.
            [a, autoInfo] = opendpd.apply(dpd, xTest.');
            testCase.verifyEqual(autoInfo.execution, 'streaming_stateful');
            testCase.verifyEqual(autoInfo.execution_requested, 'auto');
            testCase.verifyTrue(startsWith(autoInfo.execution_reason, 'auto: '));
            testCase.verifyEqual(a, s, AbsTol=single(1e-5));
            [~, namedInfo] = opendpd.apply(dpd, xTest.', Execution="streaming");
            testCase.verifyFalse(isfield(namedInfo, 'execution_reason'));
            testCase.verifyError(@() opendpd.apply(dpd, xTest.', Execution="realtime"), 'MATLAB:validators:mustBeMember');
            % Take the trained models out of OpenDPD: export, then run the packages in plain MATLAB (no Python) and compare
            % with the Python evaluator on these samples, for both semantics and for a PA and a DPD run.
            folder = string(tempname);
            testCase.addTeardown(@() rmdir(folder, 's'));
            file = fullfile(folder, "dpd.opendpd.zip");
            summary = opendpd.export(dpd, file);
            testCase.verifyEqual(summary.model, 'gru');
            testCase.verifyEqual(string(summary.run_id), dpd.ID);
            testCase.verifyEqual(summary.role, 'dpd');
            model = opendpd.load(file);
            testCase.verifyTrue(opendpd.verify(model).passed);
            testCase.verifyEqual(model.SHA256, string(summary.sha256));
            testCase.verifyEqual(opendpd.apply(model, xTest.', Execution="offline_segmented"), u, AbsTol=single(1e-5));
            testCase.verifyEqual(opendpd.apply(model, xTest.'), a, AbsTol=single(1e-5));      % auto, plain MATLAB against Python
            testCase.verifyEqual(opendpd.apply(model, xTest.', Execution="streaming", ChunkSamples=50), s, AbsTol=single(1e-5));
            again = fullfile(folder, "again.opendpd.zip");
            opendpd.export(dpd, again);
            testCase.verifyEqual(opendpd.internal.sha256(again), opendpd.internal.sha256(file));     % same run, same bytes
            opendpd.export(pa, fullfile(folder, "pa.opendpd.zip"));
            paModel = opendpd.load(fullfile(folder, "pa.opendpd.zip"));
            testCase.verifyTrue(opendpd.verify(paModel).passed);
            testCase.verifyEqual(opendpd.apply(paModel, xTest.'), opendpd.apply(pa, xTest.'), AbsTol=single(1e-5));
            [~, paInfo] = opendpd.apply(paModel, xTest.');
            testCase.verifyEqual(paInfo.output_role, 'modeled_pa_output');
            testCase.verifyError(@() opendpd.export(dpd, folder), ?MException);          % a directory is not a package file
            resumed = opendpd.getRun(p, dpd.ID);
            record = opendpd.status(resumed);
            testCase.verifyEqual(record.status, 'succeeded');
            exported = opendpd.wait(opendpd.runDPD(resumed));
            report = opendpd.result(exported);
            testCase.verifyEqual(report.evidence_type, 'dpd_surrogate');
        end

        function importMAT(testCase)
            p = testCase.Project;
            path = string(tempname) + ".mat";
            cleanup = onCleanup(@() delete(path)); %#ok<NASGU>
            x = complex(randn(2048,1), randn(2048,1)) / 8;
            y = 0.8 * x;
            save(path, 'x', 'y', '-v7');
            ds = opendpd.importMAT(p, path, SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128, ...
                Origin="synthetic", Name="mat-file-test");
            notes = jsondecode(ds.notes);
            testCase.verifyEqual(notes.source.format, 'mat');
            testCase.verifyEqual(strlength(notes.source.sha256), 64);
            [~, base, extension] = fileparts(path);
            testCase.verifyEqual(notes.source.name, char(base + extension));
            testCase.verifyEqual(notes.source.variables.input.class, 'double');
            testCase.verifyTrue(notes.source.variables.input.complex);
            testCase.verifyEqual(notes.scaling, 'none');
        end

        function importMATAnyVersionAndLayout(testCase)
            p = testCase.Project;
            folder = string(tempname); mkdir(folder);
            cleanup = onCleanup(@() rmdir(folder, 's')); %#ok<NASGU>
            rng(3);
            tx = single(randn(2048, 1)) / 8;                    % real vector: an I-only signal
            rx = 0.9 * tx;
            iq = single(randn(2048, 2)) / 8;                    % real N-by-2: I and Q columns
            out = 0.9 * iq;
            v73 = fullfile(folder, "capture v73.mat");          % HDF5-based, which SciPy cannot read
            save(v73, 'tx', 'rx', 'iq', 'out', '-v7.3');
            ds = opendpd.importMAT(p, v73, InputVariable="tx", OutputVariable="rx", SampleRate=80e6, ...
                Bandwidth=20e6, SegmentSamples=128, Origin="synthetic", Name="mat-v73-real");
            testCase.verifyEqual(ds.n_samples, 2048);
            ds = opendpd.importMAT(p, v73, InputVariable="iq", OutputVariable="out", SampleRate=80e6, ...
                Bandwidth=20e6, SegmentSamples=128, Origin="synthetic", Name="mat-v73-iq");
            testCase.verifyEqual(ds.n_samples, 2048);
            testCase.verifyEqual(jsondecode(ds.notes).source.variables.input.size(:).', [2048 2]);
            testCase.verifyError(@() opendpd.importMAT(p, v73, InputVariable="tx", OutputVariable="missing", ...
                SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128), 'opendpd:MATFile');
            cells = {1, 2}; save(fullfile(folder, "cells.mat"), 'cells', 'tx');
            testCase.verifyError(@() opendpd.importMAT(p, fullfile(folder, "cells.mat"), InputVariable="tx", ...
                OutputVariable="cells", SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128), 'opendpd:MATFile');
        end

        function deviceAutoFollowsTheServiceDefault(testCase)
            p = testCase.Project;
            rng(5);
            x = complex(single(randn(2048,1)), single(randn(2048,1))) / 8;
            ds = opendpd.importIQ(p, x, 0.8 * x, SampleRate=80e6, Bandwidth=20e6, SegmentSamples=128, ...
                Origin="synthetic", Name="device-auto");
            job = opendpd.trainPA(p, ds, Model="mp_ls", ModelParameters=struct('K', 3, 'Q', 4));
            opendpd.wait(job);
            module = py.importlib.import_module('opendpd.sdk.matlab');
            configuration = jsondecode(char(module.encode(job.Backend.config())));
            testCase.verifyEqual(configuration.execution.device, 'cpu');      % least squares never leaves the CPU
            testCase.verifyFalse(isfield(configuration.execution, 'num_threads') && ~isempty(configuration.execution.num_threads));
        end
    end
end
