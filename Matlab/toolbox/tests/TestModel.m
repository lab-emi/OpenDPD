classdef TestModel < matlab.unittest.TestCase
    % opendpd.load / verify / apply / opendpd.Model on the committed opendpd-model-v1 packages (tests/data).
    % Pure MATLAB: none of this needs Python, a project, a server or another toolbox (comm.DPD is used for one optional
    % cross-check). The packages are produced by scripts/make_matlab_model_fixtures.py; tests/unit/test_model_export.py
    % checks that they still agree with today's PyTorch code, and the golden vector inside each package is what OpenDPD's
    % evaluator produced, so these tests compare the MATLAB kernels with the evaluator without running it.
    properties (TestParameter)
        fixture = struct('gmp', 'gmp-dpd', 'gmp_ls', 'gmp_ls-dpd', 'gru', 'gru-dpd', 'gru_pa', 'gru-pa', ...
            'mp_ls', 'mp_ls-dpd', 'tres_gru', 'tres_gru-dpd')
        streamer = struct('gru', 'gru-dpd', 'gru_pa', 'gru-pa', 'gmp', 'gmp-dpd')
        plain = struct('tres_gru', 'tres_gru-dpd', 'mp_ls', 'mp_ls-dpd', 'gmp_ls', 'gmp_ls-dpd')
        notMemoryPolynomial = struct('gmp', 'gmp-dpd', 'gmp_ls', 'gmp_ls-dpd', 'gru', 'gru-dpd', 'tres_gru', 'tres_gru-dpd')
    end

    methods (TestClassSetup)
        function toolboxUnderTest(testCase)
            PackageTools.assertToolboxUnderTest(testCase);
        end
    end

    methods (Test)
        % ----- the kernels equal the evaluator ---------------------------------------------------------------------
        function packageVerifiesAgainstItsGoldenVector(testCase, fixture)
            model = opendpd.load(PackageTools.path(fixture));
            report = opendpd.verify(model);
            testCase.verifyTrue(report.passed);
            testCase.verifyEqual(report.model, char(extractBefore(string(fixture), '-')));
            testCase.verifyEqual(report.samples, 320);
            testCase.verifyLessThanOrEqual(report.tolerance_abs, 1e-5);
            testCase.verifyLessThanOrEqual(report.offline_max_abs_error, report.tolerance_abs);
            if model.Manifest.execution.streaming_stateful.available
                testCase.verifyLessThanOrEqual(report.streaming_max_abs_error, report.tolerance_abs);
            else
                testCase.verifyTrue(isnan(report.streaming_max_abs_error));
            end
        end

        function theGoldenVectorDetectsAFaultInAnyWeight(testCase, fixture)
            % Mutation test of the test: shift every weight array, then every single weight, and require the golden
            % vector to notice. Without this, "verify passed" could mean "the vector never touched that term".
            file = PackageTools.path(fixture);
            package = opendpd.internal.readPackage(file);
            names = fieldnames(package.weights);
            total = 0;
            missed = 0;
            for k = 1:numel(names)
                w = package.weights.(names{k});
                shifted = package;
                shifted.weights.(names{k}) = w + 1e-3;
                testCase.verifyFalse(opendpd.Model(shifted, file).verify().passed, ...
                    sprintf('%s: shifting %s by 1e-3 was not detected', fixture, names{k}));
                for i = 1:numel(w)
                    bumped = package;
                    v = w;
                    v(i) = v(i) + 1e-2 * (1 + abs(v(i)));
                    bumped.weights.(names{k}) = v;
                    total = total + 1;
                    missed = missed + opendpd.Model(bumped, file).verify().passed;
                end
            end
            % A weight on a feature that is tiny at the golden amplitude (|x|^3) can sit just under the tolerance; more than
            % 1% undetected would mean the golden vector does not exercise the model.
            testCase.verifyLessThanOrEqual(missed, ceil(0.01 * total), ...
                sprintf('%s: %d of %d single-weight changes of 1%% went unnoticed', fixture, missed, total));
        end

        function verificationNeverPassesOnNaN(testCase)
            file = PackageTools.path('gmp-dpd');
            package = opendpd.internal.readPackage(file);
            package.weights.gmp_weight(3) = NaN;
            report = opendpd.Model(package, file).verify();
            testCase.verifyFalse(report.passed);
            testCase.verifyEqual(report.offline_max_abs_error, Inf);
        end

        function aPackageCannotAskForALooserGoldenTest(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            golden = PackageTools.readArrays(parts, 'golden/golden.npz');
            golden.output_offline_segmented = golden.output_offline_segmented + single(1e-3);
            parts = PackageTools.replaceArrays(parts, 'golden/golden.npz', golden);
            parts = PackageTools.editManifest(parts, '("tolerance_abs": )[^,\n]+', '$1 1');
            model = opendpd.load(PackageTools.write(testCase, parts));
            testCase.verifyEqual(model.Manifest.golden.tolerance_abs, 1);        % what the package asks for
            report = model.verify();
            testCase.verifyEqual(report.tolerance_abs, 1e-5);                   % what the toolbox applies
            testCase.verifyFalse(report.passed);
            testCase.verifyGreaterThan(report.offline_max_abs_error, 5e-4);
        end

        function aPackageMayAskForAStricterGoldenTest(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            parts = PackageTools.editManifest(parts, '("tolerance_abs": )[^,\n]+', '$1 1e-9');
            report = opendpd.load(PackageTools.write(testCase, parts)).verify();
            testCase.verifyEqual(report.tolerance_abs, 1e-9);
            testCase.verifyEqual(report.passed, report.offline_max_abs_error <= 1e-9);
        end

        % ----- execution semantics -----------------------------------------------------------------------------------
        function offlineSegmentsAreIndependentAndTheTailIsTrimmed(testCase, fixture)
            model = opendpd.load(PackageTools.path(fixture));
            n = model.Manifest.signal.nperseg;
            x = PackageTools.signal(2 * n + 5, 1);
            whole = opendpd.apply(model, x);
            parts = [opendpd.apply(model, x(1:n)); opendpd.apply(model, x(n+1:2*n)); opendpd.apply(model, x(2*n+1:end))];
            testCase.verifyEqual(whole, parts, AbsTol=1e-7);
            testCase.verifySize(whole, [numel(x), 1]);
            testCase.verifyClass(whole, 'single');
            testCase.verifyFalse(isreal(whole));
            testCase.verifyGreaterThan(max(abs(whole - single(x))), 1e-3);       % not an identity in disguise
        end

        function inputOrientationPrecisionAndRealVectors(testCase)
            model = opendpd.load(PackageTools.path('mp_ls-dpd'));
            x = PackageTools.signal(300, 2);
            column = opendpd.apply(model, x);
            testCase.verifyEqual(opendpd.apply(model, x.'), column);                          % row vector
            testCase.verifyEqual(opendpd.apply(model, single(x)), column);                    % the evaluator rounds to single
            testCase.verifyEqual(opendpd.apply(model, real(x)), opendpd.apply(model, complex(real(x), 0)));   % I only
        end

        function invalidInputIsRefused(testCase)
            model = opendpd.load(PackageTools.path('mp_ls-dpd'));
            for bad = {NaN(8, 1), complex(1, Inf), zeros(0, 1), ones(4, 2), 'abc'}
                testCase.verifyError(@() opendpd.apply(model, bad{1}), ?MException, ...
                    'Invalid input was accepted');
            end
            testCase.verifyError(@() opendpd.apply(model, [1; NaN]), 'opendpd:InvalidIQ');
            testCase.verifyError(@() opendpd.apply(model, ones(4, 2)), 'opendpd:InvalidIQ');
        end

        function streamingIsIndependentOfChunking(testCase, streamer)
            model = opendpd.load(PackageTools.path(streamer));
            x = PackageTools.signal(2 * model.Manifest.signal.nperseg + 5, 3);
            whole = opendpd.apply(model, x, Execution="streaming");
            for chunk = [1 7 100 1000]
                testCase.verifyEqual(opendpd.apply(model, x, Execution="streaming_stateful", ChunkSamples=chunk), whole, ...
                    sprintf('chunk %d', chunk), AbsTol=2e-6);
            end
        end

        function streamingCarriesStateAcrossSegments(testCase, streamer)
            model = opendpd.load(PackageTools.path(streamer));
            n = model.Manifest.signal.nperseg;
            x = PackageTools.signal(3 * n, 4);
            offline = opendpd.apply(model, x);
            streamed = opendpd.apply(model, x, Execution="streaming");
            testCase.verifyEqual(streamed(1:n), offline(1:n), AbsTol=2e-6);     % the first segment starts from zero state in both
            testCase.verifyGreaterThan(max(abs(streamed(n+1:end) - offline(n+1:end))), 1e-6);
        end

        function systemObjectStreamsLikeApply(testCase, streamer)
            model = opendpd.load(PackageTools.path(streamer));
            x = PackageTools.signal(2 * model.Manifest.signal.nperseg + 5, 5);
            expected = opendpd.apply(model, x, Execution="streaming");
            actual = [model(x(1:50)); model(x(51:120)); model(x(121:end))];
            testCase.verifyEqual(actual, expected, AbsTol=2e-6);
            reset(model);
            testCase.verifyEqual(model(x), expected, AbsTol=2e-6);
            release(model);
            testCase.verifyEqual(model(x(1:10)), expected(1:10), AbsTol=2e-6);
        end

        function modelsWithoutAStreamingVariantAreRefusedNotApproximated(testCase, plain)
            model = opendpd.load(PackageTools.path(plain));
            x = PackageTools.signal(200, 6);
            testCase.verifyFalse(model.Manifest.execution.streaming_stateful.available);
            testCase.verifyError(@() opendpd.apply(model, x, Execution="streaming"), 'opendpd:NoStreamingVariant');
            testCase.verifyError(@() model(x), 'opendpd:NoStreamingVariant');
        end

        function applyReportsWhatProducedTheOutput(testCase)
            [~, info] = opendpd.apply(opendpd.load(PackageTools.path('gru-dpd')), PackageTools.signal(200, 7));
            testCase.verifyEqual(info.execution, 'offline_segmented');
            testCase.verifyEqual(info.output_role, 'predistorted_pa_input');
            testCase.verifyEqual(info.segment_samples, 128);
            [~, pa] = opendpd.apply(opendpd.load(PackageTools.path('gru-pa')), PackageTools.signal(200, 7));
            testCase.verifyEqual(pa.output_role, 'modeled_pa_output');
            [~, stream] = opendpd.apply(opendpd.load(PackageTools.path('gru-dpd')), PackageTools.signal(200, 7), Execution="streaming");
            testCase.verifyEqual(stream.execution, 'streaming_stateful');
            testCase.verifyEmpty(stream.segment_samples);
        end

        % ----- the link to MathWorks' memory polynomial ------------------------------------------------------------
        function memoryPolynomialCoefficientsAreCommDPDCoefficients(testCase)
            model = opendpd.load(PackageTools.path('mp_ls-dpd'));
            [coefficients, info] = model.commCoefficients();
            testCase.verifySize(coefficients, [4 3]);                  % memory depth Q by degree K
            testCase.verifyEqual(info.degree, 3);
            testCase.verifyEqual(info.memory_depth, 4);
            testCase.verifyTrue(contains(info.note, 'zero initial state'));
            testCase.assumeTrue(license('test', 'Communication_Toolbox') && ~isempty(which('comm.DPD')), ...
                'Communications Toolbox not available');
            x = PackageTools.signal(model.Manifest.signal.nperseg, 8);
            dpd = comm.DPD('PolynomialType', 'Memory polynomial', 'Coefficients', coefficients);
            expected = dpd(double(single(x)));
            testCase.verifyEqual(double(opendpd.apply(model, x)), expected, ...
                'one OpenDPD segment equals one comm.DPD stream with a zero initial state', AbsTol=2e-6);
        end

        function memoryPolynomialCoefficientsRunInAnRfPAmemoryOnAZeroPaddedSegment(testCase)
            model = opendpd.load(PackageTools.path('mp_ls-dpd'));
            [coefficients, info] = model.commCoefficients();
            testCase.verifySubstring(info.rf_pamemory, 'zeros(Q-1,1)');
            testCase.assumeTrue(license('test', 'RF_Toolbox') && ~isempty(which('rf.PAmemory')), 'RF Toolbox not available');
            Q = info.memory_depth;
            x = PackageTools.signal(model.Manifest.signal.nperseg, 9);
            pa = rf.PAmemory(Model='Memory polynomial', CoefficientMatrix=coefficients);
            y = pa([zeros(Q - 1, 1); double(single(x))]);
            reference = double(opendpd.apply(model, x));
            testCase.verifyLessThan(max(abs(y(Q:end) - reference)) / max(abs(reference)), 1e-6, ...
                'rf.PAmemory on a zero-padded segment equals the OpenDPD segment (registered budget: 1e-6, single interface)');
            % the delay line of rf.PAmemory starts with the first sample, so without the pad the first Q-1 outputs differ
            bare = rf.PAmemory(Model='Memory polynomial', CoefficientMatrix=coefficients);
            first = bare(double(single(x)));
            testCase.verifyGreaterThan(max(abs(first(1:Q - 1) - reference(1:Q - 1))), 1e-3);
        end

        function otherModelsHaveNoCommDPDEquivalent(testCase, notMemoryPolynomial)
            model = opendpd.load(PackageTools.path(notMemoryPolynomial));
            testCase.verifyError(@() model.commCoefficients(), 'opendpd:NoMathWorksEquivalent');
        end

        % ----- the object ------------------------------------------------------------------------------------------
        function loadedModelsDescribeThemselves(testCase)
            file = PackageTools.path('gru-dpd');
            model = opendpd.load(file);
            testCase.verifyEqual(model.Manifest.format, 'opendpd-model-v1');
            testCase.verifyEqual(model.Source, string(file));
            testCase.verifyEqual(model.SHA256, string(PackageTools.sha256(PackageTools.readBytes(file))));
            text = evalc('disp(model)');
            testCase.verifySubstring(text, 'gru');
            testCase.verifySubstring(text, 'streaming: true');
            testCase.verifyEqual(opendpd.load(string(file)).SHA256, model.SHA256);
        end

        function cloneAndSaveLoadKeepTheModel(testCase, streamer)
            model = opendpd.load(PackageTools.path(streamer));
            testCase.verifyTrue(opendpd.verify(clone(model)).passed);
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            saved = fullfile(folder, 'model.mat');
            save(saved, 'model');
            back = load(saved);
            testCase.verifyTrue(opendpd.verify(back.model).passed);
            testCase.verifyEqual(back.model.SHA256, model.SHA256);
        end

        function anEmptyModelIsRefused(testCase)
            empty = opendpd.Model();
            testCase.verifyError(@() opendpd.apply(empty, 1), 'opendpd:Package');
            testCase.verifyError(@() opendpd.verify(empty), 'opendpd:Package');
        end

        function loadNeedsAnExistingFile(testCase)
            testCase.verifyError(@() opendpd.load(fullfile(tempdir, 'does-not-exist.opendpd.zip')), 'MATLAB:validators:mustBeFile');
        end
    end

end
