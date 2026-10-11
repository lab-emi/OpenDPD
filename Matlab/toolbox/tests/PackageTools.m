classdef PackageTools
    % Helpers for the opendpd-model-v1 tests: read a committed package apart, change it, and write it (or a hostile one)
    % back as a zip, all without the toolbox code under test. The NPY/NPZ writers here are independent of
    % opendpd.internal.readNpy, so reading what they write is a real check of that parser.
    methods (Static)
        function assertToolboxUnderTest(testCase)
            % The tests must exercise the toolbox next to them, not another copy that happens to be earlier on the path.
            expected = fileparts(fileparts(mfilename('fullpath')));
            actual = fileparts(fileparts(which('opendpd.load')));
            testCase.assertEqual(actual, expected, 'opendpd.load resolves to a different copy of the toolbox than these tests');
        end

        function file = path(name)
            file = fullfile(fileparts(mfilename('fullpath')), 'data', [char(name) '.opendpd.zip']);
        end

        function x = signal(n, seed)
            stream = RandStream('twister', Seed=seed);
            x = 0.2 * complex(randn(stream, n, 1), randn(stream, n, 1));
        end

        function bytes = readBytes(file)
            fid = fopen(file, 'r');
            closeFile = onCleanup(@() fclose(fid));
            bytes = fread(fid, Inf, '*uint8');
        end

        function writeBytes(file, bytes)
            fid = fopen(file, 'w');
            closeFile = onCleanup(@() fclose(fid));
            fwrite(fid, bytes, 'uint8');
        end

        function hash = sha256(bytes)
            digest = java.security.MessageDigest.getInstance('SHA-256');
            digest.update(bytes);
            hash = lower(reshape(dec2hex(typecast(int8(digest.digest()), 'uint8'), 2).', 1, []));
        end

        function deleteIfPresent(file)
            if isfile(file)
                delete(file);
            end
        end

        % ----- zip files ---------------------------------------------------------------------------------------------
        function buildZip(file, names, payloads)
            % Java's ZipOutputStream writes whatever names it is given (including "../x"), which is the point here.
            archive = java.util.zip.ZipOutputStream(java.io.FileOutputStream(file));
            closeArchive = onCleanup(@() archive.close());
            for k = 1:numel(names)
                archive.putNextEntry(java.util.zip.ZipEntry(char(names(k))));
                data = payloads{k};
                if ~isempty(data)
                    archive.write(data(:).', 0, numel(data));
                end
                archive.closeEntry();
            end
        end

        function bytes = zipBytes(names, payloads)
            file = [tempname '.zip'];
            cleanup = onCleanup(@() PackageTools.deleteIfPresent(file));
            PackageTools.buildZip(file, names, payloads);
            bytes = PackageTools.readBytes(file);
        end

        function parts = unpack(zipFile)
            % The six files of a known-good package as {names, bytes}; tests edit them and write new archives.
            folder = tempname;
            mkdir(folder);
            removeFolder = onCleanup(@() rmdir(folder, 's'));
            unzip(zipFile, folder);
            names = ["manifest.json", "README.md", "weights.mat", "weights.npz", "golden/golden.mat", "golden/golden.npz"];
            parts = struct('names', names, 'bytes', {cell(size(names))});
            for k = 1:numel(names)
                parts.bytes{k} = PackageTools.readBytes(fullfile(folder, strrep(names(k), "/", filesep)));
            end
        end

        function file = write(testCase, parts)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            file = fullfile(folder, 'package.opendpd.zip');
            PackageTools.buildZip(file, parts.names, parts.bytes);
        end

        function parts = withEntry(parts, name, bytes)
            % Append an entry; a name that is already there makes a second entry of that name.
            parts.names(end+1) = name;
            parts.bytes{end+1} = bytes;
        end

        function parts = withoutEntry(parts, name)
            keep = parts.names ~= name;
            parts.names = parts.names(keep);
            parts.bytes = parts.bytes(keep);
        end

        function parts = replaceBytes(parts, name, bytes)
            % Replace one file and update its SHA-256 in the manifest, so that only the check under test can refuse it.
            k = find(parts.names == string(name), 1);
            parts.bytes{k} = bytes;
            pattern = ['("' regexptranslate('escape', char(name)) '": ")[0-9a-f]{64}'];
            parts = PackageTools.editManifest(parts, pattern, ['$1' PackageTools.sha256(bytes)]);
        end

        function parts = editManifest(parts, pattern, replacement)
            k = find(parts.names == "manifest.json", 1);
            text = regexprep(char(parts.bytes{k}(:).'), pattern, replacement);
            parts.bytes{k} = uint8(text(:));
        end

        function arrays = readArrays(parts, name)
            % The arrays of an .npz entry of PARTS, read with the toolbox's own parser.
            file = [tempname '.npz'];
            cleanup = onCleanup(@() PackageTools.deleteIfPresent(file));
            PackageTools.writeBytes(file, parts.bytes{parts.names == string(name)});
            arrays = opendpd.internal.readNpz(file);
        end

        function parts = replaceArrays(parts, name, arrays)
            % Replace an .npz entry (and its hash in the manifest) by one holding the fields of the struct ARRAYS.
            fields = string(fieldnames(arrays)).';
            payloads = cellfun(@(f) PackageTools.npy(arrays.(f)), cellstr(fields), UniformOutput=false);
            parts = PackageTools.replaceBytes(parts, name, PackageTools.zipBytes(fields + ".npy", payloads));
        end

        function bytes = patchDeclaredSize(bytes, entryName, newSize)
            % Overwrite the uncompressed size that the zip's central directory declares for ENTRYNAME.
            signature = uint8([80 75 1 2]);
            found = false;
            for i = 1:numel(bytes) - 46
                if isequal(bytes(i:i+3).', signature)
                    nameLength = double(typecast(bytes(i+28:i+29), 'uint16'));
                    if strcmp(char(bytes(i+46:i+45+nameLength).'), entryName)
                        bytes(i+24:i+27) = typecast(uint32(newSize), 'uint8');
                        found = true;
                        break
                    end
                end
            end
            assert(found, 'central directory entry not found');
        end

        function bytes = renameInZip(bytes, oldName, newName)
            % Rename entries in both the local and the central headers; the names must have the same length.
            assert(strlength(oldName) == strlength(newName));
            text = strrep(char(bytes(:).'), char(oldName), char(newName));
            bytes = uint8(text(:));
        end

        % ----- NumPy arrays ------------------------------------------------------------------------------------------
        function bytes = npy(array, options)
            % The .npy bytes of a real or complex single/double MATLAB array, in NumPy's C order.
            arguments
                array
                options.Descr string = ""
                options.Shape double = []
                options.Fortran (1,1) logical = false
                options.Extra string = ""
                options.Raw uint8 = uint8([])
            end
            descr = options.Descr;
            if descr == ""
                descr = "<" + ternary(isa(array, 'single'), 'f4', 'f8');
                if ~isreal(array)
                    descr = "<" + ternary(isa(array, 'single'), 'c8', 'c16');
                end
            end
            shape = options.Shape;
            if isempty(shape)
                shape = size(array);
                if isvector(array) && ~isscalar(array)
                    shape = numel(array);
                elseif isscalar(array)
                    shape = [];
                end
            end
            if isempty(options.Raw)
                isComplex = ~isreal(array);            % decided here: indexing narrows a complex array with zero imaginary part
                data = array;
                if numel(shape) > 1 && ~options.Fortran
                    data = permute(array, ndims(array):-1:1);
                end
                data = data(:);
                if isComplex
                    data = reshape([real(data).'; imag(data).'], [], 1);
                end
                raw = typecast(data, 'uint8');
            else
                raw = options.Raw(:);
            end
            digits = arrayfun(@(d) string(sprintf('%.0f', d)), shape);
            switch numel(shape)
                case 0, shapeText = "()";
                case 1, shapeText = "(" + digits + ",)";
                otherwise, shapeText = "(" + strjoin(digits, ", ") + ")";
            end
            fortran = ternary(options.Fortran, 'True', 'False');
            header = sprintf("{'descr': '%s', 'fortran_order': %s, 'shape': %s, %s}", descr, fortran, shapeText, options.Extra);
            total = 10 + strlength(header) + 1;
            pad = mod(-total, 64);
            header = header + string(repmat(' ', 1, pad)) + newline;
            bytes = [uint8([147 78 85 77 80 89 1 0]).'; typecast(uint16(strlength(header)), 'uint8').'; uint8(char(header)).'; raw(:)];
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
