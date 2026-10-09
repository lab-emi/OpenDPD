classdef TestFixedPoint < matlab.unittest.TestCase
    % A fixed-point-v1 deployment package run by plain MATLAB, bit for bit.
    %   * the six golden vectors of two real packages (the default specification and one that differs in every format) are
    %     reproduced exactly, in outputs and in the state after every sample;
    %   * the kernel equals an independent integer implementation of the specification on random formats and weights;
    %   * a package is data: hostile or damaged ones are refused with a named error, and a golden vector that was altered is
    %     reported at the sample and the signal where it differs;
    %   * the kernel builds with MATLAB Coder and the MEX function reproduces the golden vector.
    % Pure MATLAB except the Coder test; no Python.
    properties (TestParameter)
        package = struct('default', 'gru-pa', 'custom', 'gru-pa-custom')
        seed = num2cell(1:24)
        badPath = struct('parent', "../opendpd_escape.txt", 'absolute', "/tmp/opendpd_escape.txt", ...
            'inner_dotdot', "golden/../weights.json", 'directory', "weights.json/", 'windows', "C:\opendpd_escape.txt", ...
            'trailing_newline', "golden/normal/x.i16" + newline, 'extra_c_file', "c/evil.c", 'extra_golden_file', "golden/normal/extra.bin", 'bad_case_name', "golden/Bad Case/x.i16", ...
            'long_case_name', "golden/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/x.i16", 'newline', "weights.json" + newline)
        shortGolden = struct('inputs', "x.i16", 'outputs', "y.i16", 'trace', "h_trace.i16", 'final', "h_final.i16")
        modified = struct('weights', "weights.json", 'spec', "spec.json", 'readme', "README.md", 'c_source', "c/gru_fixed.c", ...
            'c_header', "c/gru_fixed.h", 'harness', "c/harness.c", 'golden_input', "golden/normal/x.i16", ...
            'golden_output', "golden/extreme/y.i16", 'golden_trace', "golden/saturation/h_trace.i16", ...
            'golden_meta', "golden/all_zero/meta.json")
        badSpec = struct( ...
            'other_specification', {{'"spec_id": "fixed-point-v1"', '"spec_id": "fixed-point-v9"'}}, ...
            'other_execution', {{'"model_key": "gru_stream"', '"model_key": "gru_offline"'}}, ...
            'other_rounding_rule', {{'round half up', 'round half down'}}, ...
            'other_saturation_rule', {{'every stored quantity saturates', 'every stored quantity wraps'}}, ...
            'input_word_too_wide', {{'("x": \{\s*"bits": )16', '$1 17'}}, ...
            'pre_activation_too_wide', {{'("pre": \{\s*"bits": )32', '$1 33'}}, ...
            'accumulator_beyond_double', {{'"accumulator_bits": 48', '"accumulator_bits": 60'}}, ...
            'weights_too_wide', {{'"weight_bits": 16', '"weight_bits": 17'}}, ...
            'table_index_fraction', {{'("sigmoid": \{\s*"function": "sigmoid",\s*"range": 8.0,\s*"index_frac": )8', '$1 17'}}, ...
            'table_range_not_whole_entries', {{'"range": 8.0', '"range": 7.3'}}, ...
            'sigmoid_table_declared_as_tanh', {{'"function": "sigmoid"', '"function": "tanh"'}})
        badWeights = struct( ...
            'other_gate_order', {{'"gate_order": \[\s*"r",\s*"z",\s*"n"\s*\]', '"gate_order": ["z", "r", "n"]'}}, ...
            'three_inputs', {{'"inputs": 2', '"inputs": 3'}}, ...
            'bias_fraction', {{'"bias": 20', '"bias": 19'}}, ...
            'weight_fraction_out_of_range', {{'"w_ih": 15,', '"w_ih": 63,'}}, ...
            'weights_for_another_specification', {{'"spec_id": "fixed-point-v1"', '"spec_id": "fixed-point-v2"'}}, ...
            'hidden_size_disagrees', {{'"hidden": 6', '"hidden": 7'}}, ...
            'a_weight_that_is_not_whole', {{'("w_ih": \[\s*\[\s*)(-?\d+)', '$1$2.5'}}, ...
            'a_weight_beyond_its_word', {{'("w_hh": \[\s*\[\s*)(-?\d+)', '$1 32768'}})
    end

    methods (TestClassSetup)
        function toolboxUnderTest(testCase)
            PackageTools.assertToolboxUnderTest(testCase);
        end
    end

    methods (Test)
        % ----- the golden vectors ------------------------------------------------------------------------------------
        function goldenVectorsAreReproducedBitForBit(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            testCase.verifyClass(model, 'opendpd.FixedModel');
            report = opendpd.verify(model);
            testCase.verifyTrue(report.passed);
            testCase.verifyEqual(report.status, 'bit_exact');
            testCase.verifyEqual(report.cases_checked, 6);
            testCase.verifyEqual(report.samples_checked, sum([model.Manifest.golden.n_samples]));
            testCase.verifyEmpty(report.mismatch_case);
            testCase.verifyEmpty(report.mismatch_sample);
            testCase.verifyEqual(report.package_c99_status, 'bit_exact');
            testCase.verifyEqual(report.spec_id, 'fixed-point-v1');
        end

        function theTwoPackagesDifferInEveryFormatTheReaderMustTakeFromThePackage(testCase)
            a = opendpd.load(FixedTools.path('gru-pa')).Manifest.spec;
            b = opendpd.load(FixedTools.path('gru-pa-custom')).Manifest.spec;
            testCase.verifyNotEqual(a.x, b.x);
            testCase.verifyNotEqual(a.h, b.h);
            testCase.verifyNotEqual(a.y, b.y);
            testCase.verifyNotEqual(b.x.frac, b.y.frac);                 % scaling the output by the input's fraction would show
            testCase.verifyNotEqual(b.x.frac, b.h.frac);
            testCase.verifyNotEqual(b.y.frac, b.h.frac);
            testCase.verifyNotEqual(a.sigmoid.value, b.sigmoid.value);
            testCase.verifyNotEqual(a.pre, b.pre);
            testCase.verifyNotEqual(a.weight_bits, b.weight_bits);
            testCase.verifyNotEqual(a.accumulator_bits, b.accumulator_bits);
            testCase.verifyNotEqual(a.sigmoid.index_frac, b.sigmoid.index_frac);
            testCase.verifyNotEqual(a.tanh.index_frac, b.tanh.index_frac);
            testCase.verifyNotEqual(a.sigmoid.range, b.sigmoid.range);
            testCase.verifyNotEqual(a.tanh.range, b.tanh.range);
        end

        function runReproducesTheGoldenIntegersThroughThePublicInterface(testCase, package)
            file = FixedTools.path(package);
            model = opendpd.load(file);
            read = opendpd.internal.readFixedPackage(file);
            hidden = model.Manifest.hidden_size;
            for entry = reshape(model.Manifest.golden, 1, [])
                g = read.golden.(entry.case_id);
                n = entry.n_samples;
                resets = reshape(entry.resets_at, 1, []) + 1;
                [yq, state, trace] = model.runInteger(reshape(g.x, 2, n).', ResetAt=resets);
                testCase.verifyEqual(yq, reshape(g.y, 2, n).', "outputs of " + entry.case_id);
                testCase.verifyEqual(trace, reshape(g.h_trace, hidden, n).', "state steps of " + entry.case_id);
                testCase.verifyEqual(state, g.h_final, "final state of " + entry.case_id);
            end
        end

        function theResultDoesNotDependOnHowTheStreamIsCutIntoChunks(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            read = opendpd.internal.readFixedPackage(FixedTools.path(package));
            n = 718;
            x = reshape(read.golden.normal.x, 2, n).';
            whole = model.runInteger(x);
            stream = RandStream('twister', Seed=5);
            chunked = zeros(size(whole), 'like', whole);
            state = [];
            first = 1;
            while first <= n
                last = min(n, first + randi(stream, 90));
                [chunked(first:last, :), state] = model.runInteger(x(first:last, :), State=state);
                first = last + 1;
            end
            testCase.verifyEqual(chunked, whole);
            [~, final] = model.runInteger(x);
            testCase.verifyEqual(state, final);
            waveform = complex(double(x(:, 1)), double(x(:, 2))) / 2^model.Manifest.spec.x.frac;
            testCase.verifyEqual(model.apply(waveform, ChunkSamples=37), model.apply(waveform));
        end

        function theStateIsCarriedAndOnlyAResetClearsIt(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            read = opendpd.internal.readFixedPackage(FixedTools.path(package));
            x = reshape(read.golden.normal.x, 2, 718).';
            block = x(1:100, :);
            first = model.runInteger(block);
            carried = model.runInteger([block; block]);
            testCase.verifyEqual(carried(1:100, :), first);
            testCase.verifyNotEqual(carried(101:200, :), first, 'the second block must see the state the first one left');
            reset = model.runInteger([block; block], ResetAt=101);
            testCase.verifyEqual(reset(101:200, :), first);
            % a reset before sample 50 makes the rest of the run what a fresh run of those samples is; one before sample 1 changes nothing
            [~, ~, resumed] = model.runInteger(block, ResetAt=[1 50]);
            [~, ~, fresh] = model.runInteger(block(50:end, :));
            testCase.verifyEqual(resumed(50:end, :), fresh);
            [~, ~, plain] = model.runInteger(block);
            testCase.verifyEqual(resumed(1:49, :), plain(1:49, :));
            testCase.verifyNotEqual(resumed(50:end, :), plain(50:end, :));
        end

        % ----- the operators ------------------------------------------------------------------------------------------
        function rescaleRoundsHalfUpAndShiftsLeftExactly(testCase)
            rescale = @opendpd.runtime.fixedRescale;
            testCase.verifyEqual(rescale([5 -5 6 -6 7 -7], 2, 0), [1 -1 2 -1 2 -2]);       % /4: 1.25 -1.25 1.5 -1.5 1.75 -1.75
            testCase.verifyEqual(rescale([0 1 -1 2 -2], 1, 0), [0 1 0 1 -1]);                % /2: 0 .5 -.5 1 -1, halves go up
            testCase.verifyEqual(rescale(3, 0, 4), 48);
            testCase.verifyEqual(rescale([-3 0 3], 5, 5), [-3 0 3]);
            stream = RandStream('twister', Seed=3);
            for k = 1:2000
                % left shifts of at most 10 bits keep the int64 oracle below its own saturation (2^63)
                v = round((rand(stream) * 2 - 1) * 2^(1 + floor(rand(stream) * 46)));
                from = floor(rand(stream) * 41);
                to = max(0, from + floor(rand(stream) * 51) - 40);
                testCase.verifyEqual(rescale(v, from, to), double(FixedTools.rescale(int64(v), from, to)), ...
                    sprintf('rescale(%d, %d, %d)', v, from, to));
            end
        end

        function inputsAreQuantisedHalfAwayFromZeroAndSaturated(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));                      % 16 bits, 14 fractional
            half = 2^-15;
            testCase.verifyEqual(model.quantiseInput(complex([1 -1 2.5 0.00003 -2.5 2].', [0 0 0 0 -3 -2].')), ...
                [16384 0; -16384 0; 32767 0; 0 0; -32768 -32768; 32767 -32768]);
            testCase.verifyEqual(model.quantiseInput(complex([half -half 3 * half -3 * half 0.5 * half].', zeros(5, 1))), ...
                [1 0; -1 0; 2 0; -2 0; 0 0]);
            % the input is rounded to single first: this double is just below half an LSB, its single is exactly half
            testCase.verifyEqual(model.quantiseInput(complex(2^-15 - 1e-13, 0)), [1 0]);
            testCase.verifyEqual(model.quantiseInput(complex(-(2^-15 - 1e-13), 0)), [-1 0]);
            custom = opendpd.load(FixedTools.path('gru-pa-custom'));              % 14 bits, 12 fractional
            testCase.verifyEqual(custom.quantiseInput(complex([1 -1 2.5 -3 2^-13 -2^-13].', zeros(6, 1))), ...
                [4096 0; -4096 0; 8191 0; -8192 0; 1 0; -1 0]);
        end

        function applyScalesTheIntegerResultByTheOutputFormat(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            stream = RandStream('twister', Seed=8);
            x = 0.4 * complex(randn(stream, 300, 1), randn(stream, 300, 1));
            [y, info] = opendpd.apply(model, x);
            testCase.verifyClass(y, 'single');
            testCase.verifySize(y, [300 1]);
            yq = double(model.runInteger(model.quantiseInput(x)));
            testCase.verifyEqual(double([real(y) imag(y)]), yq / 2^model.Manifest.spec.y.frac, AbsTol=0, RelTol=0);
            testCase.verifyEqual(info.execution, 'streaming_stateful');
            testCase.verifyEqual(info.spec_id, 'fixed-point-v1');
            testCase.verifyEqual(opendpd.apply(model, single(x)), y);
            testCase.verifyEqual(opendpd.apply(model, x, Execution="streaming"), y);
            testCase.verifyEqual(opendpd.apply(model, x, Execution="streaming_stateful", ChunkSamples=64), y);
        end

        function aFixedModelHasNoOfflineExecution(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            testCase.verifyError(@() opendpd.apply(model, ones(10, 1), Execution="offline_segmented"), 'opendpd:NoOfflineVariant');
        end

        % ----- the kernel against an independent implementation -------------------------------------------------------
        function theKernelEqualsAnIndependentIntegerImplementationOnRandomFormats(testCase, seed)
            stream = RandStream('twister', Seed=seed);
            f = FixedTools.randomFormat(stream, seed > 12);                  % half of the seeds make every sum saturate
            n = 150;
            x = round((rand(stream, n, 2) * 2 - 1) * 2^(f.xBits - 1) * 1.3);          % beyond full scale: the input saturates
            extreme = rand(stream, n, 1) < 0.1;
            x(extreme, :) = repmat([f.xMax f.xMin], nnz(extreme), 1);
            resets = sort(randperm(stream, n, 3));
            h0 = round((rand(stream, f.hidden, 1) * 2 - 1) * f.hMax);
            [y, h, trace] = opendpd.runtime.fixedGruRun(x, h0, resets, f);
            [yr, hr, tr] = FixedTools.referenceRun(f, x, h0, resets);
            testCase.verifyEqual(y, yr);
            testCase.verifyEqual(h, hr);
            testCase.verifyEqual(trace, tr);
            testCase.verifyLessThanOrEqual(max(abs(y(:))), max(f.yMax, -f.yMin));
        end

        function anAccumulatorThatFillsItsWidthRaisesAnError(testCase)
            f = FixedTools.randomFormat(RandStream('twister', Seed=2));
            f.accBits = 20;
            f.accLimit = 2^19;
            % each of the three dot products in turn: input times weights, state times weights, new state times output weights
            inputs = f; inputs.wIH(:) = 2^14;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun(repmat(f.xMax, 4, 2), zeros(f.hidden, 1), zeros(1, 0), inputs), ...
                'opendpd:FixedOverflow');
            recurrent = f; recurrent.wHH(:) = 2^14;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun(zeros(4, 2), repmat(f.hMax, f.hidden, 1), zeros(1, 0), ...
                recurrent), 'opendpd:FixedOverflow');
            output = f; output.wOut(:) = 2^14;
            output.sigmoid.values(:) = 0;                                  % z = 0: the new state is the candidate, at its maximum
            output.tanh.values(:) = output.hMax;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun(zeros(4, 2), zeros(f.hidden, 1), zeros(1, 0), output), ...
                'opendpd:FixedOverflow');
            quiet = f; quiet.accBits = 48; quiet.accLimit = 2^47;           % the same weights fit a wide enough accumulator
            quiet.wIH(:) = 2^14;
            [y, ~, ~] = opendpd.runtime.fixedGruRun(repmat(f.xMax, 4, 2), zeros(f.hidden, 1), zeros(1, 0), quiet);
            testCase.verifySize(y, [4 2]);
        end

        function theLimitOfAnAccumulatorIsAnOverflowAndOneLessIsNot(testCase)
            f = FixedTools.randomFormat(RandStream('twister', Seed=6));
            f.accBits = 20;
            f.accLimit = 2^19;
            f.xMin = -2^15; f.xMax = 2^15 - 1; f.hMin = -2^15; f.hMax = 2^15 - 1;
            f.wIH(:) = 0; f.wHH(:) = 0; f.wOut(:) = 0;                      % each case below switches on one dot product only
            none = zeros(1, 0);
            quiet = zeros(f.hidden, 1);
            input = f; input.wIH(1, :) = [2^9 0];
            testCase.verifyError(@() opendpd.runtime.fixedGruRun([2^10 0], quiet, none, input), 'opendpd:FixedOverflow');
            testCase.verifySize(opendpd.runtime.fixedGruRun([2^10 - 1 0], quiet, none, input), [1 2]);
            recurrent = f; recurrent.wHH(1, 1) = 2^9;
            state = quiet; state(1) = 2^10;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun([0 0], state, none, recurrent), 'opendpd:FixedOverflow');
            state(1) = 2^10 - 1;
            testCase.verifySize(opendpd.runtime.fixedGruRun([0 0], state, none, recurrent), [1 2]);
            output = f; output.wOut(1, 1) = 2^11;
            output.sigmoid.values(:) = 0;                                  % z = 0: the new state is the candidate itself
            output.tanh.values(:) = 2^8;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun([0 0], quiet, none, output), 'opendpd:FixedOverflow');
            output.tanh.values(:) = 2^8 - 1;
            testCase.verifySize(opendpd.runtime.fixedGruRun([0 0], quiet, none, output), [1 2]);
        end

        function theOutputPreActivationSaturatesBeforeItIsRescaledToTheOutputFormat(testCase)
            % The output format reaches +-2048 and the pre-activation only +-2: a sum that is not saturated first gives a
            % larger output than the specification does.
            f = FixedTools.randomFormat(RandStream('twister', Seed=4), true);
            f.yBits = 16; f.yFrac = 4; f.yMin = -2^15; f.yMax = 2^15 - 1;
            f.hMin = -2^15; f.hMax = 2^15 - 1; f.hFrac = 8; f.fOut = 4;
            f.sigmoid.values(:) = 0;
            f.tanh.values(:) = 100;                                         % the new state is 100 in every unit
            f.wOut = abs(f.wOut) + 1;
            f.bOut(:) = f.preMax;
            x = zeros(5, 2);
            y = opendpd.runtime.fixedGruRun(x, zeros(f.hidden, 1), zeros(1, 0), f);
            testCase.verifyEqual(y, 32 * ones(5, 2), 'floor((2^23 - 1 + 2^17) / 2^18) = 32');
            testCase.verifyEqual(y, FixedTools.referenceRun(f, x, zeros(f.hidden, 1), zeros(1, 0)));
        end

        % ----- verification finds the fault ----------------------------------------------------------------------------
        function verifyLocatesTheFirstDifferingSampleAndSignal(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            outputOnly = FixedTools.editGolden(parts, "state_reset", "y.i16", 2 * 99 + 1, 1);       % sample 100, I of the output
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, outputOnly)));
            testCase.verifyFalse(report.passed);
            testCase.verifyEqual(report.status, 'mismatch');
            testCase.verifyEqual(report.mismatch_case, 'state_reset');
            testCase.verifyEqual(report.mismatch_sample, 100);
            testCase.verifyEqual(report.mismatch_signal, 'y');
            testCase.verifyEqual(report.cases_checked, 4);
            stateOnly = FixedTools.editGolden(parts, "extreme", "h_trace.i16", 6 * 49 + 3, -1);     % sample 50, unit 3 of the state
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, stateOnly)));
            testCase.verifyEqual([string(report.mismatch_case) string(report.mismatch_signal)], ["extreme" "h"]);
            testCase.verifyEqual(report.mismatch_sample, 50);
            testCase.verifyEqual(report.cases_checked, 1);
            % both differ at one sample: the state comes first, because the output is computed from it
            both = FixedTools.editGolden(stateOnly, "extreme", "y.i16", 2 * 49 + 2, 1);
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, both)));
            testCase.verifyEqual(report.mismatch_signal, 'h');
            testCase.verifyEqual(report.mismatch_sample, 50);
            % a state that differs later than an output is reported at the output
            later = FixedTools.editGolden(parts, "saturation", "h_trace.i16", 6 * 200 + 1, 1);
            later = FixedTools.editGolden(later, "saturation", "y.i16", 2 * 30 + 1, 1);
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, later)));
            testCase.verifyEqual([string(report.mismatch_signal) string(report.mismatch_case)], ["y" "saturation"]);
            testCase.verifyEqual(report.mismatch_sample, 31);
        end

        % ----- the archive ---------------------------------------------------------------------------------------------
        function unexpectedEntriesAreRefusedAndNeverExtracted(testCase)
            parts = PackageTools.withEntry(FixedTools.unpack(FixedTools.path('gru-pa')), "evil.m", uint8('disp(''pwned'')'));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageContents');
            testCase.verifyEmpty(which('evil'));
        end

        function pathsOutsideThePackageAreRefused(testCase, badPath)
            escape = fullfile(tempdir, 'opendpd_escape.txt');
            PackageTools.deleteIfPresent(escape);
            parts = PackageTools.withEntry(FixedTools.unpack(FixedTools.path('gru-pa')), badPath, uint8('x'));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageContents');
            testCase.verifyFalse(isfile(escape));
            testCase.verifyFalse(isfile('/tmp/opendpd_escape.txt'));
        end

        function aModifiedFileIsRefused(testCase, modified)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            k = find(parts.names == modified);
            parts.bytes{k}(end) = bitxor(parts.bytes{k}(end), uint8(1));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageHash');
        end

        function aFileTheManifestDoesNotListIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editManifest(parts, '\s*"c/harness.c": "[0-9a-f]{64}",', '');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageHash');
        end

        function aFileTheManifestListsButTheArchiveLacksIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, PackageTools.withoutEntry(parts, "c/harness.c"))), ...
                'opendpd:PackageHash');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, PackageTools.withoutEntry(parts, "weights.json"))), ...
                'opendpd:PackageContents');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, PackageTools.withoutEntry(parts, "manifest.json"))), ...
                'opendpd:PackageContents');
        end

        function duplicateEntriesAreRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            twin = parts.bytes{parts.names == "c/gru_fixed.h"};
            parts = PackageTools.withEntry(parts, "c/gru_fixed.x", twin);
            file = FixedTools.write(testCase, parts);
            PackageTools.writeBytes(file, PackageTools.renameInZip(PackageTools.readBytes(file), "c/gru_fixed.x", "c/gru_fixed.h"));
            testCase.verifyError(@() opendpd.load(file), 'opendpd:PackageContents');
        end

        function anEntryThatInflatesBeyondItsDeclaredSizeIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.replaceBytes(parts, "golden/normal/meta.json", zeros(3e6, 1, 'uint8'));
            file = FixedTools.write(testCase, parts);
            testCase.assertLessThan(dir(file).bytes, 1e6, 'the test archive should be small');
            lying = fullfile(fileparts(file), 'lying.fixed-point-v1.zip');
            PackageTools.writeBytes(lying, PackageTools.patchDeclaredSize(PackageTools.readBytes(file), ...
                'golden/normal/meta.json', 1000));
            testCase.verifyError(@() opendpd.load(lying), 'opendpd:PackageContents');
        end

        function sizeLimitsAreEnforced(testCase)
            file = FixedTools.path('gru-pa');
            testCase.verifyError(@() opendpd.internal.readFixedPackage(file, MaxEntryBytes=20000), 'opendpd:PackageContents');
            testCase.verifyError(@() opendpd.internal.readFixedPackage(file, MaxTotalBytes=50000), 'opendpd:PackageContents');
            package = opendpd.internal.readFixedPackage(file);
            testCase.verifyEqual(package.manifest.spec.spec_id, 'fixed-point-v1');
        end

        function aFileThatIsNotAPackageIsRefused(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            text = fullfile(folder, 'not-a-zip.zip');
            PackageTools.writeBytes(text, uint8('this is not a zip file'));
            testCase.verifyError(@() opendpd.load(text), 'opendpd:Package');
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            for edit = {'"fixed-point-v1"', '"fixed-point-v2"'; '^.*$', 'not json'}.'
                changed = FixedTools.editManifest(parts, edit{1}, edit{2});
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, changed)), 'opendpd:Package');
            end
            json = FixedTools.replaceBytes(parts, "weights.json", uint8('{"hidden": '));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, json)), 'opendpd:Package');
        end

        function aGoldenCaseNameThatCannotBeAFieldNameIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            for name = ["golden/1abc/x.i16", "golden/a-b/x.i16"]
                extra = PackageTools.withEntry(parts, name, uint8([1 0]));
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, extra)), 'opendpd:PackageContents');
            end
            % a valid-looking name that is a MATLAB keyword is listed everywhere, so only the field name stops it
            renamed = parts;
            renamed.names = replace(renamed.names, "golden/normal/", "golden/end/");
            renamed = FixedTools.editManifest(renamed, '"golden/normal/', '"golden/end/');
            renamed = FixedTools.editManifest(renamed, '"case_id": "normal"', '"case_id": "end"');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, renamed)), 'opendpd:PackageContents');
        end

        % ----- what the package says --------------------------------------------------------------------------------------
        function aSpecificationThisToolboxDoesNotImplementIsRefused(testCase, badSpec)
            parts = FixedTools.editSpec(FixedTools.unpack(FixedTools.path('gru-pa')), badSpec{1}, badSpec{2});
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aSpecificationThatDiffersFromTheManifestsIsRefused(testCase)
            % a difference nothing else would notice: a 47-bit accumulator loads fine on its own
            parts = FixedTools.editJson(FixedTools.unpack(FixedTools.path('gru-pa')), "spec.json", ...
                '"accumulator_bits": 48', '"accumulator_bits": 47');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
            alone = FixedTools.editSpec(FixedTools.unpack(FixedTools.path('gru-pa')), '"accumulator_bits": 48', '"accumulator_bits": 47');
            testCase.verifyEqual(opendpd.verify(opendpd.load(FixedTools.write(testCase, alone))).status, 'bit_exact');
        end

        function aGoldenFileThatIsShorterThanItsIndexSaysIsRefused(testCase, shortGolden)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            name = "golden/all_zero/" + shortGolden;
            bytes = parts.bytes{parts.names == name};
            parts = FixedTools.replaceBytes(parts, name, bytes(1:end - 2));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function manifestFieldsThatAreShownOrReportedAreChecked(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            runId = regexp(FixedTools.textOf(parts, "manifest.json"), '"run_id": "([^"]*)"', 'tokens', 'once');
            pattern = ['"run_id": "' regexptranslate('escape', runId{1}) '"'];
            for unsafe = ["has space", "esc\\u001b[2Jx", "../up", "new\\nline"]
                changed = FixedTools.editManifest(parts, pattern, ['"run_id": "' char(unsafe) '"']);
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, changed)), 'opendpd:Package', char(unsafe));
            end
            verdict = FixedTools.editManifest(parts, '"status": "bit_exact"', '"status": "perfect"');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, verdict)), 'opendpd:Package');
            size = FixedTools.editManifest(parts, '"hidden_size": 6', '"hidden_size": 7');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, size)), 'opendpd:Package');
        end

        function weightsThatAreNotWhatTheSpecificationDescribesAreRefused(testCase, badWeights)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editJson(parts, "weights.json", badWeights{1}, badWeights{2});
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aPackageWhoseIntegersCouldLeaveTheExactRangeOfDoublePrecisionIsRefused(testCase)
            % Weights and inputs with no fractional bits, scaled up by 20 bits to the pre-activation format: a 48-bit
            % accumulator could then reach 2^67. Everything else in the package is consistent, so only the bound can refuse it.
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editJson(parts, "weights.json", '"w_ih": 15,', '"w_ih": 0,');
            parts = FixedTools.editManifest(parts, ...
                '("name": "w_ih",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )15', '$1 0');
            parts = FixedTools.editSpec(parts, '("x": \{\s*"bits": 16,\s*"frac": )14', '$1 0');
            file = FixedTools.write(testCase, parts);
            try
                opendpd.load(file);
                message = '';
            catch cause
                testCase.verifyEqual(cause.identifier, 'opendpd:Package');
                message = cause.message;
            end
            testCase.verifySubstring(message, 'exactly');
        end

        function aManifestWithoutTheFileHashesIsNotAPackage(testCase)
            parts = FixedTools.editManifest(FixedTools.unpack(FixedTools.path('gru-pa')), '"files":', '"file_hashes":');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aTableOfTheWrongLengthIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editJson(parts, "weights.json", '("tanh_table": \[\s*)(-?\d+),', '$1');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aManifestThatDisagreesAboutTheQuantisationIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editManifest(parts, '("name": "w_hh",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )15', '$1 14');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function goldenVectorsThatDoNotMatchTheirIndexAreRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            lie = FixedTools.editManifest(parts, '("case_id": "all_zero",\s*"description": "[^"]*",\s*"n_samples": )256', '$1 257');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, lie)), 'opendpd:Package');
            reset = FixedTools.editManifest(parts, '("resets_at": \[\s*0,\s*256,\s*512,\s*)768', '$1 5000');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, reset)), 'opendpd:Package');
            missing = FixedTools.editManifest(parts, '"case_id": "long_sequence"', '"case_id": "longer_sequence"');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, missing)), 'opendpd:Package');
            torn = FixedTools.editGolden(parts, "normal", "h_final.i16", 1, 1);
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, torn)), 'opendpd:Package');
        end

        % ----- the interface -----------------------------------------------------------------------------------------------
        function loadChoosesTheReaderFromTheEntryNames(testCase)
            testCase.verifyClass(opendpd.load(FixedTools.path('gru-pa')), 'opendpd.FixedModel');
            testCase.verifyClass(opendpd.load(PackageTools.path('gru-dpd')), 'opendpd.Model');
            testCase.verifyEqual(opendpd.internal.packageKind(FixedTools.path('gru-pa')), "fixed");
            testCase.verifyEqual(opendpd.internal.packageKind(PackageTools.path('gru-dpd')), "model");
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            junk = fullfile(folder, 'junk.zip');
            PackageTools.writeBytes(junk, uint8('junk'));
            testCase.verifyEqual(opendpd.internal.packageKind(junk), "model");
            % a model package that also carries the two JSON names is still a model package, and the model reader refuses it
            parts = PackageTools.unpack(PackageTools.path('gru-dpd'));
            parts = PackageTools.withEntry(PackageTools.withEntry(parts, "spec.json", uint8('{}')), "weights.json", uint8('{}'));
            mixed = PackageTools.write(testCase, parts);
            testCase.verifyEqual(opendpd.internal.packageKind(mixed), "model");
            testCase.verifyError(@() opendpd.load(mixed), 'opendpd:PackageContents');
        end

        function aFixedModelPrintsAndAnEmptyOneRefusesToRun(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            text = evalc('disp(model)');
            testCase.verifySubstring(text, 'opendpd.FixedModel');
            testCase.verifySubstring(text, 'fixed-point-v1');
            empty = opendpd.FixedModel();
            testCase.verifyError(@() empty.apply(ones(4, 1)), 'opendpd:Package');
            testCase.verifyError(@() empty.verify(), 'opendpd:Package');
        end

        function inputsThatAreNotIntegersOrIQAreRefused(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            testCase.verifyError(@() model.runInteger([1.5 2]), 'MATLAB:validators:mustBeInteger');
            testCase.verifyError(@() model.runInteger([1 2 3]), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.runInteger(zeros(0, 2)), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.runInteger([1 NaN]), 'MATLAB:validators:mustBeFinite');
            testCase.verifyError(@() model.runInteger([1 2], State=ones(5, 1)), 'opendpd:InvalidState');
            testCase.verifyError(@() model.runInteger([1 2], State=2^20 * ones(6, 1)), 'opendpd:InvalidState');
            testCase.verifyError(@() model.runInteger([1 2], ResetAt=0), 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() model.apply([1 NaN]), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.apply([]), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.apply(ones(3, 3)), 'opendpd:InvalidIQ');
        end

        function generateCodeDoesNotAcceptAFixedModel(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            folder = fullfile(testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder, 'out');
            testCase.verifyError(@() opendpd.generateCode(model, folder), 'opendpd:CodegenFixedPoint');
            testCase.verifyFalse(isfolder(folder));
        end

        function loadingDoesNotLoadPython(testCase)
            testCase.assumeFalse(strcmp(char(pyenv().Status), 'Loaded'), 'Python is already loaded in this session');
            opendpd.verify(opendpd.load(FixedTools.path('gru-pa')));
            testCase.verifyNotEqual(char(pyenv().Status), 'Loaded');
        end

        % ----- MATLAB Coder ---------------------------------------------------------------------------------------------------
        function theKernelBuildsWithMatlabCoderAndTheMexFunctionReproducesTheGoldenVectors(testCase, package)
            testCase.assumeTrue(license('test', 'MATLAB_Coder') && ~isempty(which('codegen')), 'MATLAB Coder is not available');
            testCase.assumeNotEmpty(mex.getCompilerConfigurations('C', 'Selected'), 'no C compiler is selected for MATLAB Coder');
            file = FixedTools.path(package);
            model = opendpd.load(file);
            read = opendpd.internal.readFixedPackage(file);
            f = model.kernelFormat();
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            testCase.applyFixture(matlab.unittest.fixtures.CurrentFolderFixture(folder));
            writelines(["function [y, h, trace] = fixedEntry(x, h, resetAt, f)", ...
                "[y, h, trace] = opendpd.runtime.fixedGruRun(x, h, resetAt, f);", "end"], 'fixedEntry.m');
            configuration = coder.config('mex');
            configuration.EnableJIT = false;
            codegen('fixedEntry', '-args', {coder.typeof(0, [Inf 2]), zeros(f.hidden, 1), coder.typeof(0, [1 Inf]), ...
                coder.Constant(f)}, '-config', configuration, '-o', 'fixedEntryMex', '-silent');
            testCase.addTeardown(@() clear('fixedEntryMex'));
            for entry = reshape(model.Manifest.golden, 1, [])
                g = read.golden.(entry.case_id);
                n = entry.n_samples;
                resets = reshape(entry.resets_at, 1, []) + 1;
                [y, ~, trace] = fixedEntryMex(double(reshape(g.x, 2, n).'), zeros(f.hidden, 1), resets, f);
                testCase.verifyEqual(y, double(reshape(g.y, 2, n).'), "outputs of " + entry.case_id);
                testCase.verifyEqual(trace, double(reshape(g.h_trace, f.hidden, n).'), "state steps of " + entry.case_id);
            end
        end
    end
end
