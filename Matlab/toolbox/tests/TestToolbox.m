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
                SampleRate=80e6, Bandwidth=20e6), 'opendpd:InvalidIQ');
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
            pa = opendpd.wait(opendpd.trainPA(p, ds, ModelParameters=parameters, Training=training));
            dpd = opendpd.wait(opendpd.trainDPD(p, ds, PA=pa, ModelParameters=parameters, Training=training));
            xTest = x(1:257);
            [u, info] = opendpd.apply(dpd, xTest.');
            reference = dpd.Backend.apply(py.numpy.asarray([real(xTest), imag(xTest)]));
            referenceIQ = single(reference{1});
            testCase.verifyEqual(u, complex(referenceIQ(:,1), referenceIQ(:,2)), AbsTol=single(1e-6));
            testCase.verifySize(u, [257, 1]);
            testCase.verifyClass(u, 'single');
            testCase.verifyEqual(info.execution, 'offline_segmented');
            testCase.verifyEqual(info.output_role, 'predistorted_pa_input');
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
            ds = opendpd.importMAT(p, path, SampleRate=80e6, Bandwidth=20e6, ...
                Origin="synthetic", Name="mat-file-test");
            notes = jsondecode(ds.notes);
            testCase.verifyEqual(notes.source.format, 'mat');
            testCase.verifyEqual(notes.source_arrays.input.dtype, 'complex128');
        end
    end
end
