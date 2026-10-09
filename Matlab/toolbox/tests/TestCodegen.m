classdef TestCodegen < matlab.unittest.TestCase
    % opendpd.generateCode on the committed opendpd-model-v1 packages (tests/data): the standalone classes it writes equal
    % opendpd.apply, pass their own golden check, need nothing from the toolbox, cannot be made to run text from a package,
    % and (where MATLAB Coder, a C compiler and Simulink are installed) build as a MEX function and run in a MATLAB System
    % block in both simulation modes. Those two groups are skipped, not failed, where the products are missing.
    properties (TestParameter)
        variant = struct( ...
            'gru_offline', {{'gru-dpd', 'offline_segmented'}}, 'gru_streaming', {{'gru-dpd', 'streaming_stateful'}}, ...
            'gru_pa_offline', {{'gru-pa', 'offline_segmented'}}, 'gru_pa_streaming', {{'gru-pa', 'streaming_stateful'}}, ...
            'tres_gru_offline', {{'tres_gru-dpd', 'offline_segmented'}}, ...
            'gmp_offline', {{'gmp-dpd', 'offline_segmented'}}, 'gmp_streaming', {{'gmp-dpd', 'streaming_stateful'}}, ...
            'gmp_ls_offline', {{'gmp_ls-dpd', 'offline_segmented'}}, 'mp_ls_offline', {{'mp_ls-dpd', 'offline_segmented'}})
        offline = struct('gru', 'gru-dpd', 'gru_pa', 'gru-pa', 'tres_gru', 'tres_gru-dpd', 'gmp', 'gmp-dpd', ...
            'gmp_ls', 'gmp_ls-dpd', 'mp_ls', 'mp_ls-dpd')
        streamer = struct('gru', 'gru-dpd', 'gru_pa', 'gru-pa', 'gmp', 'gmp-dpd')
        plain = struct('tres_gru', 'tres_gru-dpd', 'mp_ls', 'mp_ls-dpd', 'gmp_ls', 'gmp_ls-dpd')
        simulation = struct('interpreted', 'Interpreted execution', 'generated', 'Code generation')
    end

    methods (TestClassSetup)
        function toolboxUnderTest(testCase)
            PackageTools.assertToolboxUnderTest(testCase);
        end
    end

    methods (Test)
        % ----- the generated class is the model -----------------------------------------------------------------
        function theGeneratedCheckPassesAndAgreesWithTheToolboxVerifier(testCase, variant)
            [name, model] = testCase.generate(variant{:});
            report = feval(name + "Check");
            verifier = model.verify();
            testCase.verifyTrue(report.passed);
            testCase.verifyEqual(report.model, char(extractBefore(string(variant{1}), '-')));
            testCase.verifyEqual(report.execution, variant{2});
            testCase.verifyEqual(report.samples, 320);
            testCase.verifyLessThanOrEqual(report.tolerance_abs, 1e-5);
            testCase.verifyLessThanOrEqual(report.max_abs_error, report.tolerance_abs);
            if strcmp(variant{2}, 'offline_segmented')
                reference = verifier.offline_max_abs_error;
            else
                reference = verifier.streaming_max_abs_error;
            end
            testCase.verifyEqual(report.max_abs_error, reference, AbsTol=1e-12);
        end

        function offlineClassesEqualApplyOnWaveformsOfAnyLength(testCase, offline)
            [name, model] = testCase.generate(offline, 'offline_segmented');
            segment = model.Manifest.signal.nperseg;
            obj = feval(name);
            for count = [1, 2, segment - 1, segment, segment + 1, 3 * segment + 5]
                x = PackageTools.signal(count, count);
                y = obj(x);
                testCase.verifyClass(y, 'single');
                testCase.verifySize(y, [count 1]);
                testCase.verifyFalse(isreal(y));
                testCase.verifyEqual(double(y), double(opendpd.apply(model, x, Execution="offline_segmented")), ...
                    sprintf('%s, %d samples', offline, count), AbsTol=1e-12);
            end
        end

        function streamingClassesEqualApplyWhateverTheChunkSizes(testCase, streamer)
            [name, model] = testCase.generate(streamer, 'streaming_stateful');
            x = PackageTools.signal(500, 3);
            reference = double(opendpd.apply(model, x, Execution="streaming"));
            obj = feval(name);
            y = complex(zeros(0, 1));
            first = 1;
            for chunk = [1, 2, 3, 37, 100, 1, 211, 500]
                if first > numel(x)
                    break
                end
                last = min(first + chunk - 1, numel(x));
                y = [y; double(obj(x(first:last)))]; %#ok<AGROW>
                first = last + 1;
            end
            testCase.verifyEqual(numel(y), numel(x));
            testCase.verifyEqual(y, reference, AbsTol=1e-12);
            % reset returns the object to the start of a stream
            reset(obj);
            testCase.verifyEqual(double(obj(x)), reference, AbsTol=1e-12);
            testCase.verifyNotEqual(double(obj(x)), reference, 'a second pass without reset must continue from the state');
        end

        function inputsOfEveryAcceptedKindGiveAComplexSingleColumn(testCase)
            [name, model] = testCase.generate('gru-dpd', 'offline_segmented');
            x = PackageTools.signal(150, 8);
            reference = double(opendpd.apply(model, x, Execution="offline_segmented"));
            obj = feval(name);
            for input = {single(x), x, x.', single(x).'}
                release(obj);
                y = obj(input{1});
                testCase.verifyClass(y, 'single');
                testCase.verifySize(y, [150 1]);
                testCase.verifyEqual(double(y), reference, AbsTol=1e-12);
            end
            release(obj);
            fromReal = obj(single(real(x)));
            testCase.verifyEqual(double(fromReal), double(opendpd.apply(model, single(real(x)), Execution="offline_segmented")), AbsTol=1e-12);
        end

        function badInputsAreRefusedLikeApplyRefusesThem(testCase)
            [name, ~] = testCase.generate('gru-dpd', 'offline_segmented');
            obj = feval(name);
            testCase.verifyError(@() obj([]), 'MATLAB:expectedVector');
            testCase.verifyError(@() obj(zeros(0, 1)), 'MATLAB:expectedNonempty');
            testCase.verifyError(@() obj(ones(3, 3)), 'MATLAB:expectedVector');
            testCase.verifyError(@() obj([1; NaN]), 'MATLAB:expectedFinite');
            testCase.verifyError(@() obj([1; Inf]), 'MATLAB:expectedFinite');
            testCase.verifyError(@() obj('abc'), 'MATLAB:invalidType');
            testCase.verifyError(@() obj(int8([1 2 3])), 'MATLAB:invalidType');
            testCase.verifyError(@() obj(true(3, 1)), 'MATLAB:invalidType');
        end

        function theGeneratedCheckDetectsAFaultInTheGeneratedCode(testCase)
            % Mutation test of the check: damage the class text in a way that matters, and require the check to notice.
            faults = {
                'gru-dpd', 'streaming_stateful', 'h = (1 - z) .* c + z .* h;', 'h = (1 - z) .* h + z .* c;'
                'gru-dpd', 'offline_segmented', 'projected = weightIH * input.'' + biasIH;', 'projected = weightIH * input.'';'
                'gru-dpd', 'offline_segmented', 'out = out2 * obj.FcWeight.'' + obj.FcBias.'';', 'out = out2 * obj.FcWeight.'';'
                'gru-pa', 'offline_segmented', 'z = 1 ./ (1 + exp(-(gi(hidden+1:2*hidden) + gh(hidden+1:2*hidden))));', 'z = 1 ./ (1 + exp(-(gi(hidden+1:2*hidden))));'
                'tres_gru-dpd', 'offline_segmented', 'next = circshift(iq, -1, 1);', 'next = [iq(2:end, :); 0 0];'
                'tres_gru-dpd', 'offline_segmented', 'shift = 16 * (j - 1);', 'shift = 15 * (j - 1);'
                'gmp-dpd', 'streaming_stateful', 'history = block(numel(block) - {KEEP} + (1:{KEEP}));', 'history = block(numel(block) - {KEEP} + (1:{KEEP})) * 0;'
                'gmp-dpd', 'offline_segmented', 'ap = abs([zeros(M - 1, 1); xp]);', 'ap = abs(xp);'
                'mp_ls-dpd', 'offline_segmented', 'coefficients(k * Q + q + 1)', 'coefficients(q * K + k + 1)'
                'gmp_ls-dpd', 'offline_segmented', 'd = delaySignal(x, l);', 'd = delaySignal(x, l + 1);'};
            for i = 1:size(faults, 1)
                [name, model, result] = testCase.generate(faults{i, 1}, faults{i, 2});
                keep = '0';
                if strcmp(model.Manifest.model.key, 'gmp')
                    keep = num2str(2 * (model.codegenInputs().Kernel.M - 1));
                end
                pattern = strrep(faults{i, 3}, '{KEEP}', keep);
                text = fileread(result.Files(1));
                testCase.assertEqual(count(text, pattern), 1, ['pattern not found once: ' pattern]);
                writeText(result.Files(1), strrep(text, pattern, strrep(faults{i, 4}, '{KEEP}', keep)));
                try
                    detected = ~feval(name + "Check").passed;       % a wrong answer, or no answer at all
                catch
                    detected = true;
                end
                testCase.verifyTrue(detected, ['fault not detected: ' pattern]);
            end
        end

        % ----- what the generated files depend on -------------------------------------------------------------------
        function theGeneratedFilesNeedNothingFromTheToolbox(testCase, variant)
            [name, ~, result] = testCase.generate(variant{:});
            for k = 1:3
                [files, products] = matlab.codetools.requiredFilesAndProducts(char(result.Files(k)));
                testCase.verifyEqual(string({products.Name}), "MATLAB");
                testCase.verifyTrue(all(startsWith(string(files), string(result.Folder) + filesep)), ...
                    'a generated file depends on a file outside its own folder');
            end
            for file = result.Files(1:3).'
                for line = splitlines(string(fileread(file))).'
                    stripped = strip(line);
                    if ~startsWith(stripped, "%")
                        testCase.verifyFalse(contains(stripped, "opendpd."), ...
                            "code line refers to the toolbox: " + stripped);
                    end
                end
            end
            report = withoutToolbox(testCase, @() feval(name + "Check"));
            testCase.verifyTrue(report.passed);
        end

        function generationIsDeterministicAndIndependentOfTheFolder(testCase, variant)
            model = opendpd.load(PackageTools.path(variant{1}));
            first = opendpd.generateCode(model, testCase.newFolder(), Name="Deterministic", Execution=variant{2});
            second = opendpd.generateCode(model, testCase.newFolder(), Name="Deterministic", Execution=variant{2});
            for k = 1:numel(first.Files)
                testCase.verifyEqual(PackageTools.readBytes(first.Files(k)), PackageTools.readBytes(second.Files(k)), ...
                    char(first.Files(k)));
            end
        end

        function generatedFilesAreAsciiWithUnixLineEndings(testCase, variant)
            [~, ~, result] = testCase.generate(variant{:});
            for file = result.Files.'
                bytes = PackageTools.readBytes(file);
                testCase.verifyTrue(all(bytes < 128), char(file));
                testCase.verifyFalse(any(bytes == 13), char(file));
                testCase.verifyEqual(bytes(end), uint8(10), char(file));
            end
        end

        function theResultNamesTheFilesAndTheModel(testCase)
            model = opendpd.load(PackageTools.path('gru-dpd'));
            folder = testCase.newFolder();
            result = opendpd.generateCode(model, folder, Name="NamedResult");
            testCase.verifyEqual(result.Name, 'NamedResult');
            testCase.verifyEqual(result.Execution, 'offline_segmented');
            testCase.verifyEqual(result.Model, 'gru');
            testCase.verifyEqual(result.PackageSHA256, char(model.SHA256));
            testCase.verifyEqual(result.Files, [string(folder) + filesep + ["NamedResult.m"; "NamedResultStep.m"; ...
                "NamedResultCheck.m"; "README_NamedResult.md"]]);
            testCase.verifyTrue(all(isfile(result.Files)));
            % "streaming" is accepted as the short name that opendpd.apply also accepts
            other = opendpd.generateCode(model, testCase.newFolder(), Name="NamedStream", Execution="streaming");
            testCase.verifyEqual(other.Execution, 'streaming_stateful');
            % the package's own provenance is in the readme and in the class
            readme = fileread(result.Files(4));
            testCase.verifyTrue(contains(readme, char(model.SHA256)) && contains(readme, model.Manifest.run.run_id));
        end

        % ----- names, existing files, refusals ----------------------------------------------------------------------
        function namesAreCheckedBeforeAnythingIsWritten(testCase)
            model = opendpd.load(PackageTools.path('gru-dpd'));
            folder = fullfile(testCase.newFolder(), 'new');
            taken = testCase.newFolder();
            writeText(fullfile(taken, 'TakenByAUser.m'), sprintf('function TakenByAUser()\nend\n'));
            writeText(fullfile(taken, 'ShadowedStep.m'), sprintf('function ShadowedStep()\nend\n'));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(taken));
            for bad = {"1bad", "has space", "a-b", "x.y", string(repmat('x', 1, 41)), "sin", "load", "TakenByAUser", "Shadowed", "end"}
                testCase.verifyError(@() opendpd.generateCode(model, folder, Name=bad{1}), 'opendpd:CodegenName', ...
                    "name " + bad{1});
            end
            testCase.verifyFalse(isfolder(folder), 'a refused name must not create the folder');
            testCase.verifyError(@() opendpd.generateCode(model, "", Name="Anything"), 'MATLAB:validators:mustBeNonzeroLengthText');
        end

        function defaultNamesSayWhatTheModelIs(testCase)
            expected = {'gru-dpd', 'OpenDPDDpdGru'; 'gru-pa', 'OpenDPDPaGru'; 'tres_gru-dpd', 'OpenDPDDpdTresGru'; ...
                'gmp-dpd', 'OpenDPDDpdGmp'; 'gmp_ls-dpd', 'OpenDPDDpdGmpLs'; 'mp_ls-dpd', 'OpenDPDDpdMpLs'};
            for i = 1:size(expected, 1)
                model = opendpd.load(PackageTools.path(expected{i, 1}));
                result = opendpd.generateCode(model, testCase.newFolder());
                testCase.verifyEqual(result.Name, expected{i, 2});
                testCase.verifyTrue(all(isfile(result.Files)));
            end
        end

        function existingFilesAreOnlyReplacedWhenTheyWereGeneratedAndAsked(testCase)
            model = opendpd.load(PackageTools.path('gru-dpd'));
            folder = testCase.newFolder();
            first = opendpd.generateCode(model, folder, Name="Keeper");
            before = arrayfun(@(f) PackageTools.sha256(PackageTools.readBytes(f)), first.Files, UniformOutput=false);
            testCase.verifyError(@() opendpd.generateCode(model, folder, Name="Keeper"), 'opendpd:CodegenExists');
            again = opendpd.generateCode(model, folder, Name="Keeper", Overwrite=true);
            after = arrayfun(@(f) PackageTools.sha256(PackageTools.readBytes(f)), again.Files, UniformOutput=false);
            testCase.verifyEqual(after, before);
            % a file of the user's own is never replaced, and nothing else is touched when it stops the call
            writeText(first.Files(1), sprintf('%% my own notes\n'));
            testCase.verifyError(@() opendpd.generateCode(model, folder, Name="Keeper", Overwrite=true), ...
                'opendpd:CodegenForeignFile');
            testCase.verifyEqual(fileread(first.Files(1)), sprintf('%% my own notes\n'));
            testCase.verifyEqual(PackageTools.sha256(PackageTools.readBytes(first.Files(4))), before{4});
        end

        function streamingIsRefusedWhereThePackageHasNoStreamingVariant(testCase, plain)
            model = opendpd.load(PackageTools.path(plain));
            folder = fullfile(testCase.newFolder(), 'new');
            testCase.verifyError(@() opendpd.generateCode(model, folder, Execution="streaming_stateful"), ...
                'opendpd:NoStreamingVariant');
            testCase.verifyFalse(isfolder(folder));
        end

        function textInThePackageCannotBecomeCode(testCase)
            hostile = sprintf('x''; system(''echo pwned''); %%{ \nsystem(''echo pwned2'') %% %s', repmat('long', 1, 40));
            parts = PackageTools.unpack(PackageTools.path('gru-dpd'));
            for key = ["run_id", "role", "type"]
                literal = strrep(strrep(jsonencode(hostile), '\', '\\'), '$', '\$');
                pattern = ['"' char(key) '": "[^"]*"'];
                testCase.assertEqual(numel(regexp(char(parts.bytes{parts.names == "manifest.json"}(:).'), pattern)), 1);
                parts = PackageTools.editManifest(parts, pattern, ['"' char(key) '": ' literal]);
            end
            model = opendpd.load(PackageTools.write(testCase, parts));
            testCase.assertEqual(model.Manifest.run.run_id, hostile, 'the hostile text must reach the generator');
            for execution = ["offline_segmented", "streaming_stateful"]
                [name, ~, result] = testCase.generate(model, execution);
                for file = result.Files.'
                    text = string(fileread(file));
                    code = strings(0, 1);
                    for line = splitlines(text).'
                        if ~startsWith(strip(line), "%") && ~endsWith(file, ".md")
                            code(end+1, 1) = regexprep(line, '''[^'']*''', ''''''); %#ok<AGROW>
                        end
                    end
                    testCase.verifyFalse(any(contains(code, ["system", "pwned", "echo"])), char(file));
                end
                safe = regexprep(hostile, '[^A-Za-z0-9 _.:/@+=,()-]', '?');
                instance = feval(name);
                testCase.verifyEqual(instance.RunId, [safe(1:64) '...']);
                testCase.verifyTrue(feval(name + "Check").passed);
            end
        end

        function weightsAreWrittenExactlyWhateverTheirMagnitude(testCase)
            model = opendpd.load(PackageTools.path('mp_ls-dpd'));
            kernel = model.codegenInputs().Kernel;
            special = [0; -0; 1e-300; 5e-324; -2.5e-310; pi; -1/3; 1e30; -1.7e30; 2^-52; 1 - 2^-53; 1];
            count = kernel.K * kernel.Q;
            stream = RandStream('twister', Seed=5);
            part = special(mod(0:count - 1, numel(special)) + 1) .* sign(randn(stream, count, 1) + 0.1);
            values = complex(part, flipud(part) / 7);
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            parts = PackageTools.replaceArrays(parts, "weights.npz", struct('coefficients', values));
            damaged = opendpd.load(PackageTools.write(testCase, parts));
            [name, ~] = testCase.generate(damaged, 'offline_segmented');
            x = PackageTools.signal(300, 21) / 10;
            instance = feval(name);
            testCase.verifyEqual(instance(x), opendpd.apply(damaged, x, Execution="offline_segmented"), AbsTol=0);
        end

        function weightsThatAreNotFiniteAreRefused(testCase)
            parts = PackageTools.unpack(PackageTools.path('gru-dpd'));
            arrays = PackageTools.readArrays(parts, "weights.npz");
            arrays.fc_weight(1) = NaN;
            parts = PackageTools.replaceArrays(parts, "weights.npz", arrays);
            model = opendpd.load(PackageTools.write(testCase, parts));
            folder = fullfile(testCase.newFolder(), 'new');
            testCase.verifyError(@() opendpd.generateCode(model, folder), 'opendpd:Package');
            testCase.verifyFalse(isfolder(folder));
        end

        function packagesTooLargeForSourceTextAreRefused(testCase)
            memory = 200;
            degree = 14;
            parts = PackageTools.unpack(PackageTools.path('gmp-dpd'));
            text = char(parts.bytes{parts.names == "manifest.json"}(:).');
            for edit = {'"memory_length": \d+', ['"memory_length": ' num2str(memory)]; '"degree": \d+', ['"degree": ' num2str(degree)]; ...
                    '"history_samples": \d+', ['"history_samples": ' num2str(2 * (memory - 1))]}.'
                testCase.assertEqual(numel(regexp(text, edit{1})), 1, edit{1});
                parts = PackageTools.editManifest(parts, edit{1}, edit{2});
            end
            count = memory * (1 + (degree - 1) * memory);
            parts = PackageTools.replaceArrays(parts, "weights.npz", struct('gmp_weight', zeros(count, 1, 'single')));
            model = opendpd.load(PackageTools.write(testCase, parts));
            folder = fullfile(testCase.newFolder(), 'new');
            testCase.verifyError(@() opendpd.generateCode(model, folder), 'opendpd:CodegenTooLarge');
            testCase.verifyFalse(isfolder(folder));
        end

        % ----- MATLAB Coder -----------------------------------------------------------------------------------------
        function matlabCoderBuildsTheClassAndTheMexFunctionReproducesTheGoldenVector(testCase, variant)
            testCase.assumeCoder();
            [name, model] = testCase.generate(variant{:});
            golden = model.codegenInputs().Golden;
            mex = testCase.buildMex(name, coder.typeof(complex(single(0)), [Inf 1]));
            x = complex(single(golden.input(:, 1)), single(golden.input(:, 2)));
            if strcmp(variant{2}, 'offline_segmented')
                y = mex(x);
                reference = golden.output_offline_segmented;
            else
                chunk = model.Manifest.golden.streaming_chunk_samples;
                y = complex(zeros(numel(x), 1, 'single'));
                for first = 1:chunk:numel(x)
                    last = min(first + chunk - 1, numel(x));
                    y(first:last) = mex(x(first:last));
                end
                reference = golden.output_streaming_stateful;
            end
            testCase.verifyClass(y, 'single');
            worst = max(abs([real(double(y)) - double(reference(:, 1)); imag(double(y)) - double(reference(:, 2))]));
            testCase.verifyLessThanOrEqual(worst, 1e-5);
            % the compiled code and the interpreted class compute the same thing
            interpreted = feval(name + "Check");
            testCase.verifyEqual(worst, interpreted.max_abs_error, AbsTol=1e-9);
        end

        function matlabCoderAlsoBuildsFixedSizeFramesSamplesAndDoubleInputs(testCase)
            testCase.assumeCoder();
            [name, model] = testCase.generate('gru-dpd', 'streaming_stateful');
            x = PackageTools.signal(40, 2);
            reference = double(opendpd.apply(model, x, Execution="streaming"));
            frame = testCase.buildMex(name, coder.typeof(complex(single(0)), [40 1]), "frame");
            testCase.verifyEqual(double(frame(single(x))), reference, AbsTol=1e-6);
            clear(char(name + "_mex_frame"));
            sample = testCase.buildMex(name, coder.typeof(complex(0), [1 1]), "sample");
            y = zeros(40, 1);
            for k = 1:40
                y(k) = sample(x(k));
            end
            testCase.verifyEqual(y, reference, AbsTol=1e-6);
        end

        % ----- Simulink -----------------------------------------------------------------------------------------------
        function aSimulinkMatlabSystemBlockRunsTheClassOnAFrame(testCase, variant, simulation)
            testCase.assumeSimulink();
            [name, model] = testCase.generate(variant{:});
            golden = model.codegenInputs().Golden;
            x = complex(single(golden.input(:, 1)), single(golden.input(:, 2)));
            if strcmp(variant{2}, 'offline_segmented')
                reference = golden.output_offline_segmented;
            else
                reference = golden.output_streaming_stateful;
            end
            testCase.inTemporaryFolder();
            mdl = testCase.newModel();
            assignin(get_param(mdl, 'ModelWorkspace'), 'frame', x);
            add_block('simulink/Sources/Constant', [mdl '/In'], 'Value', 'frame', 'OutDataTypeStr', 'single', 'SampleTime', '1');
            add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/Block'], 'System', char(name), ...
                'SimulateUsing', simulation);
            add_block('simulink/Sinks/To Workspace', [mdl '/Out'], 'VariableName', 'yout', 'SaveFormat', 'Array');
            add_line(mdl, 'In/1', 'Block/1');
            add_line(mdl, 'Block/1', 'Out/1');
            set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', 'FixedStep', '1', 'StopTime', '0');
            out = sim(mdl);
            y = out.yout;
            testCase.verifyClass(y, 'single');
            testCase.verifySize(y, [numel(x) 1]);
            testCase.verifyFalse(isreal(y));
            worst = max(abs([real(double(y)) - double(reference(:, 1)); imag(double(y)) - double(reference(:, 2))]));
            testCase.verifyLessThanOrEqual(worst, 1e-5);
        end

        function aSimulinkTransmitChainEqualsTheMatlabChainSampleBySample(testCase, simulation)
            % DPD then PA, one sample per time step: the recurrent state has to survive from step to step.
            testCase.assumeSimulink();
            dpd = opendpd.load(PackageTools.path('gru-dpd'));
            pa = opendpd.load(PackageTools.path('gru-pa'));
            folder = testCase.newFolder();
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(folder));
            dpdName = TestCodegen.freshName();
            paName = TestCodegen.freshName();
            opendpd.generateCode(dpd, folder, Name=dpdName, Execution="streaming");
            opendpd.generateCode(pa, folder, Name=paName, Execution="streaming");
            x = single(PackageTools.signal(120, 4));
            reference = double(opendpd.apply(pa, opendpd.apply(dpd, x, Execution="streaming"), Execution="streaming"));
            testCase.inTemporaryFolder();
            mdl = testCase.newModel();
            assignin(get_param(mdl, 'ModelWorkspace'), 'samples', timeseries(x, (0:numel(x) - 1).'));
            add_block('simulink/Sources/From Workspace', [mdl '/In'], 'VariableName', 'samples', ...
                'OutputAfterFinalValue', 'Holding final value', 'SampleTime', '1');
            add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/DPD'], 'System', char(dpdName), ...
                'SimulateUsing', simulation);
            add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/PA'], 'System', char(paName), ...
                'SimulateUsing', simulation);
            add_block('simulink/Sinks/To Workspace', [mdl '/Out'], 'VariableName', 'yout', 'SaveFormat', 'Array');
            add_line(mdl, 'In/1', 'DPD/1');
            add_line(mdl, 'DPD/1', 'PA/1');
            add_line(mdl, 'PA/1', 'Out/1');
            set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', 'FixedStep', '1', ...
                'StopTime', num2str(numel(x) - 1));
            out = sim(mdl);
            testCase.verifyEqual(double(out.yout(:)), reference, AbsTol=1e-6);
        end
    end

    methods (Access = private)
        function [name, model, result] = generate(testCase, source, execution)
            % Write a class for a committed package (or an already loaded model) into a new folder that is on the path.
            if isa(source, 'opendpd.Model')
                model = source;
            else
                model = opendpd.load(PackageTools.path(source));
            end
            folder = testCase.newFolder();
            name = TestCodegen.freshName();
            result = opendpd.generateCode(model, folder, Name=name, Execution=execution);
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(folder));
            name = string(result.Name);
        end

        function folder = newFolder(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
        end

        function inTemporaryFolder(testCase)
            testCase.applyFixture(matlab.unittest.fixtures.CurrentFolderFixture(testCase.newFolder()));
        end

        function assumeCoder(testCase)
            testCase.assumeTrue(license('test', 'MATLAB_Coder') && ~isempty(which('codegen')), 'MATLAB Coder is not available');
            testCase.assumeNotEmpty(mex.getCompilerConfigurations('C', 'Selected'), 'no C compiler is selected for MATLAB Coder');
        end

        function assumeSimulink(testCase)
            testCase.assumeTrue(license('test', 'Simulink') && ~isempty(which('new_system')), 'Simulink is not available');
        end

        function mex = buildMex(testCase, name, inputType, suffix)
            % Build the Coder entry point of the generated class as a MEX function in a new current folder.
            if nargin < 4
                suffix = "";
            end
            testCase.inTemporaryFolder();
            mexName = char(name + "_mex" + ternary(suffix == "", "", "_" + suffix));
            configuration = coder.config('mex');
            configuration.EnableJIT = false;
            codegen(char(name + "Step"), '-args', {inputType}, '-config', configuration, '-o', mexName, '-silent');
            testCase.addTeardown(@() clear(mexName));
            mex = str2func(mexName);
        end

        function mdl = newModel(testCase)
            mdl = char(TestCodegen.freshName());
            new_system(mdl);
            testCase.addTeardown(@() closeModel(mdl));
        end
    end

    methods (Static, Access = private)
        function name = freshName()
            % A class name that has never been loaded in this session, so that no cached definition is reused.
            name = string(['G' erase(char(java.util.UUID.randomUUID), '-')]);
        end
    end
end

function varargout = withoutToolbox(testCase, fcn)
% Run FCN with every folder that holds the +opendpd package removed from the path and the current folder elsewhere.
originalPath = path;
originalFolder = pwd;
restore = onCleanup(@() restoreSession(originalPath, originalFolder)); %#ok<NASGU>
cd(tempdir);
for entry = strsplit(originalPath, pathsep)
    if isfolder(fullfile(entry{1}, '+opendpd'))
        rmpath(entry{1});
    end
end
testCase.assertEmpty(which('opendpd.load'), 'the toolbox is still reachable');
[varargout{1:max(nargout, 1)}] = fcn();
end

function restoreSession(originalPath, originalFolder)
path(originalPath);
cd(originalFolder);
end

function closeModel(mdl)
if bdIsLoaded(mdl)
    close_system(mdl, 0);
end
end

function writeText(file, text)
fid = fopen(file, 'w');
closeFile = onCleanup(@() fclose(fid));
fwrite(fid, text, 'char');
end

function value = ternary(condition, a, b)
if condition
    value = a;
else
    value = b;
end
end
