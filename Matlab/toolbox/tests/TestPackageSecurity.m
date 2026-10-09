classdef TestPackageSecurity < matlab.unittest.TestCase
    % A model package from somewhere else is data, never code. Every test here builds a hostile or damaged package from
    % a committed one (PackageTools) and requires opendpd.load to refuse it with a named error, to have run nothing, and
    % to have written nothing outside its own temporary folder. Pure MATLAB; no Python.
    properties (TestParameter)
        foreign = struct('function_handle', 'function_handle', 'struct', 'struct', 'cell', 'cell', 'char', 'char', ...
            'map', 'map', 'trap', 'trap')
        badPath = struct('parent', "../opendpd_escape.txt", 'absolute', "/tmp/opendpd_escape.txt", ...
            'inner_dotdot', "golden/../weights.mat", 'directory', "weights.mat/", 'windows', "C:\opendpd_escape.txt", ...
            'nested', "golden/golden.mat/x")
        modified = struct('weights_mat', "weights.mat", 'weights_npz', "weights.npz", 'golden_mat', "golden/golden.mat", ...
            'readme', "README.md")
        badNpz = struct('parent', "../x.npy", 'space', "a b.npy", 'leading_digit', "1abc.npy", 'pickle', "x.pkl", ...
            'directory', "x.npy/", 'nested', "sub/x.npy")
        arrayType = struct('single', 'single', 'double', 'double', 'single_complex', 'singlecomplex', ...
            'double_complex', 'doublecomplex')
    end

    methods (TestClassSetup)
        function toolboxUnderTest(testCase)
            PackageTools.assertToolboxUnderTest(testCase);
        end
    end

    methods (Test)
        % ----- the archive -------------------------------------------------------------------------------------------
        function unexpectedEntriesAreRefusedAndNeverExtracted(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            parts = PackageTools.withEntry(parts, "evil.m", uint8('disp(''pwned'')'));
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, parts)), 'opendpd:PackageContents');
            testCase.verifyEmpty(which('evil'));
        end

        function pathsOutsideThePackageAreRefused(testCase, badPath)
            escape = fullfile(tempdir, 'opendpd_escape.txt');
            PackageTools.deleteIfPresent(escape);
            parts = PackageTools.withEntry(PackageTools.unpack(PackageTools.path('mp_ls-dpd')), badPath, uint8('x'));
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, parts)), 'opendpd:PackageContents');
            testCase.verifyFalse(isfile(escape));
            testCase.verifyFalse(isfile('/tmp/opendpd_escape.txt'));
        end

        function duplicateEntriesAreRefused(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            % Two entries with the same name, content, size and hash: only the duplicate itself is wrong.
            parts = PackageTools.replaceBytes(parts, 'golden/golden.npz', parts.bytes{parts.names == "golden/golden.mat"});
            file = PackageTools.write(testCase, parts);
            % Java will not write two entries of one name, so rename golden.npz to golden.mat inside the finished archive
            % (same length; headers only, the compressed data is untouched). weights.npz stays, so only the duplicate is wrong.
            PackageTools.writeBytes(file, PackageTools.renameInZip(PackageTools.readBytes(file), "golden/golden.npz", "golden/golden.mat"));
            testCase.verifyError(@() opendpd.load(file), 'opendpd:PackageContents');
        end

        function aModifiedFileIsRefused(testCase, modified)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            k = find(parts.names == modified);
            parts.bytes{k}(end) = bitxor(parts.bytes{k}(end), uint8(1));
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, parts)), 'opendpd:PackageHash');
        end

        function aFileTheManifestDoesNotListIsRefused(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            parts = PackageTools.editManifest(parts, '\s*"golden/golden.npz": "[0-9a-f]{64}",', '');
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, parts)), 'opendpd:PackageHash');
        end

        function anEntryThatInflatesBeyondItsDeclaredSizeIsRefused(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            parts = PackageTools.replaceBytes(parts, 'golden/golden.mat', zeros(3e6, 1, 'uint8'));
            file = PackageTools.write(testCase, parts);
            testCase.assertLessThan(dir(file).bytes, 1e6, 'the test archive should be small');
            lying = fullfile(fileparts(file), 'lying.opendpd.zip');
            PackageTools.writeBytes(lying, PackageTools.patchDeclaredSize(PackageTools.readBytes(file), 'golden/golden.mat', 1000));
            testCase.verifyError(@() opendpd.load(lying), 'opendpd:PackageContents');
        end

        function sizeLimitsAreEnforced(testCase)
            file = PackageTools.path('mp_ls-dpd');
            % 4000 bytes is above every array inside the packages (so the .npz reader cannot be what refuses them) and below
            % the outer golden entries, which is the check this test is about.
            testCase.verifyError(@() opendpd.internal.readPackage(file, MaxEntryBytes=4000), 'opendpd:PackageContents');
            testCase.verifyError(@() opendpd.internal.readPackage(file, MaxTotalBytes=5000), 'opendpd:PackageContents');
            package = opendpd.internal.readPackage(file);                    % the defaults accept a real package
            testCase.verifyEqual(package.manifest.format, 'opendpd-model-v1');
        end

        function aFileThatIsNotAPackageIsRefused(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            text = fullfile(folder, 'not-a-zip.opendpd.zip');
            PackageTools.writeBytes(text, uint8('this is not a zip file'));
            testCase.verifyError(@() opendpd.load(text), 'opendpd:Package');
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            for edit = {'"opendpd-model-v1"', '"opendpd-model-v2"'; '^.*$', 'not json'}.'
                changed = PackageTools.editManifest(parts, edit{1}, edit{2});
                testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, changed)), 'opendpd:Package');
            end
            for missing = ["weights.npz", "manifest.json"]
                testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, PackageTools.withoutEntry(parts, missing))), ...
                    'opendpd:PackageContents');
            end
        end

        function aManifestThatNestsDeeplyOrHoldsTooMuchIsRefusedBeforeJsondecodeCanHurtMatlab(testCase)
            % jsondecode recurses once per level of nesting: 100000 levels end the MATLAB process with a segmentation fault.
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            k = find(parts.names == "manifest.json");
            for depth = [9 5000 100000]
                changed = parts;
                changed.bytes{k} = uint8([repmat('[', 1, depth) repmat(']', 1, depth)]).';
                testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:Package'), ...
                    sprintf('manifest.json nests %d levels deep', depth));
            end
            allowed = parts;                                   % eight levels are what the limit allows: refused for what it is
            allowed.bytes{k} = uint8(['{"format": "opendpd-model-v1", "x": ' repmat('[', 1, 7) repmat(']', 1, 7) '}']).';
            testCase.verifyFalse(contains(refusalMessage(testCase, allowed, 'opendpd:PackageHash'), 'levels deep'));
            many = parts;
            many.bytes{k} = uint8(['{"format": "opendpd-model-v1", "x": [' repmat('0,', 1, 30000) '0]}']).';
            testCase.verifySubstring(refusalMessage(testCase, many, 'opendpd:Package'), 'array elements');
            long = parts;                                      % jsondecode takes half a minute on a million digits
            long.bytes{k} = uint8(['{"format": "opendpd-model-v1", "x": 0.' repmat('1', 1, 1e6) '}']).';
            tic;
            testCase.verifySubstring(refusalMessage(testCase, long, 'opendpd:Package'), 'holds a number or word of');
            testCase.verifyLessThan(toc, 10);
            big = parts;
            big.bytes{k} = zeros(4e6 + 1, 1, 'uint8');
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, big)), 'opendpd:PackageContents');
        end

        function aManifestThatIsNotOneObjectIsRefusedWithAMessage(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            text = char(parts.bytes{parts.names == "manifest.json"}(:).');
            notOneObject = {['[' text ',' text ']'], '[]', '{}', '"opendpd-model-v1"', 'null', '7'};
            for replacement = notOneObject
                changed = PackageTools.replaceBytes(parts, 'manifest.json', uint8(replacement{1}).');
                testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:Package'), 'Not an opendpd-model-v1 package', ...
                    replacement{1}(1:min(end, 20)));
            end
            files = PackageTools.editManifest(parts, '("files": )(\{[^}]*\})', '$1[$2,$2]');
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, files)), 'opendpd:PackageHash');
            asList = PackageTools.editManifest(parts, '"format": "opendpd-model-v1"', '"format": ["opendpd-model-v1"]');
            testCase.verifySubstring(refusalMessage(testCase, asList, 'opendpd:Package'), 'Not an opendpd-model-v1 package');
        end

        function aManifestThatDeclaresFewerGruLayersThanTheWeightsHoldIsRefused(testCase)
            parts = PackageTools.unpack(PackageTools.path('gru-dpd'));
            testCase.assertNotEmpty(regexp(char(parts.bytes{parts.names == "manifest.json"}(:).'), '"num_layers": 2', 'once'), ...
                'the fixture must have two layers');
            fewer = PackageTools.editManifest(parts, '"num_layers": 2', '"num_layers": 1');
            message = refusalMessage(testCase, fewer, 'opendpd:Package');
            testCase.verifySubstring(message, 'which the 1-layer model of its manifest does not use');
            testCase.verifyEqual(class(opendpd.load(PackageTools.path('gru-dpd'))), 'opendpd.Model');
            % an array the model never uses under a name of the layer family is as much a mismatch as a third layer
            weights = PackageTools.readArrays(parts, 'weights.npz');
            weights.rnn_weight_ih_l5 = weights.rnn_weight_ih_l0;
            odd = PackageTools.replaceArrays(parts, 'weights.npz', weights);
            testCase.verifySubstring(refusalMessage(testCase, odd, 'opendpd:Package'), 'rnn_weight_ih_l5');
            weights = PackageTools.readArrays(parts, 'weights.npz');
            weights.rnn_gain = 1;
            other = PackageTools.replaceArrays(parts, 'weights.npz', weights);
            testCase.verifySubstring(refusalMessage(testCase, other, 'opendpd:Package'), 'rnn_gain');
        end

        function textInTheManifestIsMadePlainBeforeAnyoneSeesIt(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            changed = PackageTools.editManifest(parts, '"note": "[^"]*"', '"note": "a\\u001b[2Jb\\u0007c"');
            model = opendpd.load(PackageTools.write(testCase, changed));
            testCase.verifyEqual(model.Manifest.evidence.note, 'a?[2Jb?c');
            wrongType = PackageTools.editManifest(parts, '"format": "opendpd-model-v1"', '"format": {}');
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, wrongType)), 'opendpd:Package');
        end

        function aManifestThatDoesNotMatchTheWeightsIsRefused(testCase)
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            wrongSize = PackageTools.replaceArrays(parts, 'weights.npz', struct('coefficients', complex(ones(5, 1), 0)));
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, wrongSize)), 'opendpd:Package');
            unknown = PackageTools.editManifest(parts, '"key": "mp_ls"', '"key": "lstm"');
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, unknown)), 'opendpd:Package');
            absurd = PackageTools.editManifest(parts, '("nperseg": )[0-9]+', '$1 1e12');
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, absurd)), 'opendpd:Package');
            gru = PackageTools.unpack(PackageTools.path('gru-dpd'));
            weights = PackageTools.readArrays(gru, 'weights.npz');
            weights.fc_weight = weights.fc_weight(:, 1:end-1);
            bad = PackageTools.replaceArrays(gru, 'weights.npz', weights);
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, bad)), 'opendpd:Package');
            weights = PackageTools.readArrays(gru, 'weights.npz');
            weights = rmfield(weights, 'rnn_weight_hh_l1');
            missing = PackageTools.replaceArrays(gru, 'weights.npz', weights);
            testCase.verifyError(@() opendpd.load(PackageTools.write(testCase, missing)), 'opendpd:Package');
        end

        % ----- MAT files are never opened ---------------------------------------------------------------------------
        function theMatFilesAreHashedButNeverOpened(testCase, foreign)
            % whos('-file') and load() both run loadobj for classes on the path, so the loader must not open a MAT file
            % from a package at all. A hostile weights.mat (with a valid hash) is therefore harmless, and the
            % toolbox's numbers come from the .npz.
            switch foreign
                case 'function_handle', value = @(x) x + 1;
                case 'struct', value = struct('a', 1);
                case 'cell', value = {1, 2};
                case 'char', value = 'abc';
                case 'map', value = containers.Map();
                case 'trap', value = TrapObject();
            end
            marker = TrapObject.markerFile();
            PackageTools.deleteIfPresent(marker);
            testCase.addTeardown(@() PackageTools.deleteIfPresent(marker));
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            hostile = fullfile(folder, 'hostile.mat');
            save(hostile, 'value', '-v7');
            PackageTools.deleteIfPresent(marker);
            parts = PackageTools.unpack(PackageTools.path('mp_ls-dpd'));
            parts = PackageTools.replaceBytes(parts, 'weights.mat', PackageTools.readBytes(hostile));
            model = opendpd.load(PackageTools.write(testCase, parts));
            testCase.verifyTrue(opendpd.verify(model).passed);
            testCase.verifyFalse(isfile(marker), 'loading the package ran code from the MAT file');
            if strcmp(foreign, 'trap')
                % Controls: inspecting the same file the way one might (whos -file) does run the object's loadobj. That is
                % why the loader never does it, and it shows that the check above could have failed.
                listing = whos('-file', hostile); %#ok<NASGU>
                testCase.verifyTrue(isfile(marker), 'the trap does not fire under whos -file; this test proves nothing');
            end
        end

        % ----- the arrays: a strict NumPy reader --------------------------------------------------------------------
        function npyRoundTripsEveryTypeAndShape(testCase, arrayType)
            stream = RandStream('twister', Seed=11);
            for shape = {[1 1], [5 1], [3 4], [2 3 4], [1 7], [4 1 3]}
                if ismember(arrayType, ["singlecomplex", "doublecomplex"])
                    a = complex(randn(stream, shape{1}), randn(stream, shape{1}));
                else
                    a = randn(stream, shape{1});
                end
                a = cast(a, strrep(arrayType, 'complex', ''));
                back = opendpd.internal.readNpy(PackageTools.npy(a), 'a.npy');
                expected = a;
                if isvector(a) && ~isscalar(a)
                    expected = a(:);                                   % NumPy has no row/column: 1-D reads as a column
                end
                testCase.verifyEqual(back, expected, sprintf('shape %s', mat2str(shape{1})));
                testCase.verifyClass(back, class(expected));
            end
            fortran = reshape(1:12, 3, 4) + 0.5;
            testCase.verifyEqual(opendpd.internal.readNpy(PackageTools.npy(fortran, Fortran=true), 'f.npy'), fortran);
            testCase.verifyEqual(opendpd.internal.readNpy(PackageTools.npy(single(7)), 's.npy'), single(7));          % 0-d
        end

        function theReaderAgreesWithNumPyAndTheMatFilesOnTheCommittedPackages(testCase)
            % Two independent routes to the same numbers: NumPy wrote both files of a package; MATLAB's own MAT reader
            % (used here on our own committed files only) and the strict .npy reader must agree exactly.
            for name = ["gmp-dpd", "gmp_ls-dpd", "gru-dpd", "gru-pa", "mp_ls-dpd", "tres_gru-dpd"]
                parts = PackageTools.unpack(PackageTools.path(name));
                for pair = {'weights.npz', 'weights.mat'; 'golden/golden.npz', 'golden/golden.mat'}.'
                    npz = PackageTools.readArrays(parts, pair{1});
                    file = [tempname '.mat'];
                    PackageTools.writeBytes(file, parts.bytes{parts.names == string(pair{2})});
                    mat = load(file);
                    delete(file);
                    testCase.verifyEqual(sort(fieldnames(npz)), sort(fieldnames(mat)), char(name));
                    for field = fieldnames(mat).'
                        testCase.verifyEqual(npz.(field{1}), mat.(field{1}), sprintf('%s %s', name, field{1}));
                    end
                end
            end
        end

        function hostileNpyArraysAreRefused(testCase)
            good = PackageTools.npy(single([1 2 3 4]));
            read = @(b) opendpd.internal.readNpy(b, 'a.npy');
            refuse = @(b, why) testCase.verifyError(@() read(b), 'opendpd:PackageContents', why);
            testCase.verifyEqual(read(good), single([1 2 3 4]).');
            refuse(good(2:end), 'bad magic');
            refuse(good(1:20), 'truncated header');
            refuse([good; uint8(0)], 'extra data after the array');
            refuse(good(1:end-1), 'missing data');
            refuse(PackageTools.npy(single([1 2 3 4]), Descr="|O"), 'object array (would be unpickled by NumPy)');
            refuse(PackageTools.npy(single([1 2 3 4]), Descr="<i2"), 'integer type');
            refuse(PackageTools.npy(single([1 2 3 4]), Descr=">f4"), 'big-endian');
            refuse(PackageTools.npy(single([1 2 3 4]), Descr="[('a', '<f4')]"), 'structured type');
            refuse(PackageTools.npy(single([1 2 3 4]), Extra="'extra': 1, "), 'extra header key');
            refuse(PackageTools.npy(single([1 2 3 4]), Shape=[1e12 1e12], Raw=uint8([1 2 3 4])), 'absurd shape');
            refuse(PackageTools.npy(single([1 2 3 4]), Shape=[3 2]), 'shape that does not match the data');
            version = good;
            version(7) = 9;
            refuse(version, 'unknown format version');
            ascii = good;
            ascii(20) = 200;
            refuse(ascii, 'non-ASCII header');
            lengthLie = good;
            lengthLie(9:10) = typecast(uint16(60000), 'uint8');
            refuse(lengthLie, 'header length beyond the file');
            injected = PackageTools.npy(single([1 2 3 4]), Descr="<f4', 'x': __import__('os').system('echo pwned')#");
            refuse(injected, 'code in the header is never evaluated');
        end

        function npzEntryNamesAreRestricted(testCase, badNpz)
            file = [tempname '.npz'];
            testCase.addTeardown(@() PackageTools.deleteIfPresent(file));
            PackageTools.buildZip(file, [badNpz], {PackageTools.npy(single(1))});
            testCase.verifyError(@() opendpd.internal.readNpz(file), 'opendpd:PackageContents');
            PackageTools.buildZip(file, "ok_1.npy", {PackageTools.npy(single(1))});
            testCase.verifyEqual(opendpd.internal.readNpz(file), struct('ok_1', single(1)));
        end

        function anNpzThatInflatesBeyondItsDeclaredSizeIsRefused(testCase)
            file = [tempname '.npz'];
            testCase.addTeardown(@() PackageTools.deleteIfPresent(file));
            PackageTools.buildZip(file, "big.npy", {zeros(3e6, 1, 'uint8')});
            PackageTools.writeBytes(file, PackageTools.patchDeclaredSize(PackageTools.readBytes(file), 'big.npy', 100));
            testCase.verifyError(@() opendpd.internal.readNpz(file), 'opendpd:PackageContents');
        end

        function npzLimitsAreEnforced(testCase)
            file = [tempname '.npz'];
            testCase.addTeardown(@() PackageTools.deleteIfPresent(file));
            names = "a" + string(1:5) + ".npy";
            PackageTools.buildZip(file, names, repmat({PackageTools.npy(single(1:10))}, 1, 5));
            testCase.verifyError(@() opendpd.internal.readNpz(file, MaxEntries=4), 'opendpd:PackageContents');
            testCase.verifyError(@() opendpd.internal.readNpz(file, MaxEntryBytes=100), 'opendpd:PackageContents');
            testCase.verifyError(@() opendpd.internal.readNpz(file, MaxTotalBytes=300), 'opendpd:PackageContents');
            testCase.verifyEqual(numel(fieldnames(opendpd.internal.readNpz(file))), 5);
        end

        % ----- what the code is allowed to do -----------------------------------------------------------------------
        function theLoaderAndRuntimeLoadNothingAndNeverCallPython(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            internal = fullfile(root, '+opendpd', '+internal');
            loader = string(fullfile(internal, {'readPackage.m', 'readNpz.m', 'readNpy.m', 'copyZipEntry.m', 'sha256.m'}));
            loader = [loader, string(fullfile(root, '+opendpd', {'Model.m', 'verify.m'}))];
            runtime = dir(fullfile(root, '+opendpd', '+runtime', '*.m'));
            runtime = string(fullfile({runtime.folder}, {runtime.name}));
            forbidden = '(?<![A-Za-z0-9_.])(load|whos|matfile|eval|evalc|evalin|feval|str2func|unzip|run|system|dos|unix|urlread|webread)\s*\(';
            for file = [loader, runtime, string(fullfile(root, '+opendpd', 'load.m'))]
                text = fileread(file);
                if endsWith(file, 'load.m') && ~endsWith(file, 'Model.m')
                    text = regexprep(text, 'function model = load\(', '');         % its own name
                end
                testCase.verifyEmpty(regexp(text, forbidden, 'once'), sprintf('%s calls something that can run code', file));
                testCase.verifyEmpty(regexp(text, '(?<![A-Za-z0-9_.])py\.', 'once'), sprintf('%s refers to Python', file));
                testCase.verifyEmpty(regexp(text, '(?<![A-Za-z0-9_.])bridge\s*\(', 'once'), sprintf('%s calls the bridge', file));
            end
            testCase.verifyNotEmpty(runtime);
        end
    end
end

function message = refusalMessage(testCase, parts, identifier)
% Load PARTS as a package, check that the error has IDENTIFIER, and return its message.
message = '';
try
    opendpd.load(PackageTools.write(testCase, parts));
    testCase.verifyFail('The package was not refused.');
catch cause
    testCase.verifyEqual(cause.identifier, identifier, cause.message);
    message = cause.message;
end
end
