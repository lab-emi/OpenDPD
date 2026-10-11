classdef FixedTools
    % Helpers for the fixed-point-v1 tests: read a committed deployment package apart, change it, and write it (or a hostile
    % one) back as a zip, and an independent integer implementation of the specification to compare the kernel with. None of it
    % uses the toolbox code under test except where a test says so.
    methods (Static)
        function file = path(name)
            file = fullfile(fileparts(mfilename('fullpath')), 'data', [char(name) '.fixed-point-v1.zip']);
        end

        function parts = unpack(zipFile)
            % Every entry of a package as {names, bytes}; tests edit them and write new archives.
            archive = java.util.zip.ZipFile(char(zipFile));
            closeArchive = onCleanup(@() archive.close());
            names = strings(1, 0);
            entries = archive.entries();
            while entries.hasMoreElements()
                names(end+1) = string(entries.nextElement().getName()); %#ok<AGROW>
            end
            folder = tempname;
            mkdir(folder);
            removeFolder = onCleanup(@() rmdir(folder, 's'));
            unzip(zipFile, folder);
            parts = struct('names', names, 'bytes', {cell(size(names))});
            for k = 1:numel(names)
                parts.bytes{k} = PackageTools.readBytes(fullfile(folder, strrep(names(k), "/", filesep)));
            end
        end

        function file = write(testCase, parts)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            file = fullfile(folder, 'package.fixed-point-v1.zip');
            PackageTools.buildZip(file, parts.names, parts.bytes);
        end

        function text = textOf(parts, name)
            text = char(parts.bytes{parts.names == string(name)}(:).');
        end

        function parts = replaceBytes(parts, name, bytes)
            % Replace one file and update its SHA-256 in the manifest, so that only the check under test can refuse it.
            k = find(parts.names == string(name), 1);
            parts.bytes{k} = bytes;
            pattern = ['("' regexptranslate('escape', char(name)) '": ")[0-9a-f]{64}'];
            parts = FixedTools.editManifest(parts, pattern, ['$1' PackageTools.sha256(bytes)]);
            % a golden vector is also indexed with its hash, in the case's own object: a regenerated package has both right
            golden = regexp(char(name), '^golden/([a-z_]+)/(x|y|h_final|h_trace)\.i16$', 'tokens', 'once');
            if ~isempty(golden)
                field = struct('x', 'input', 'y', 'output', 'h_final', 'state', 'h_trace', 'trace').(golden{2});
                parts = FixedTools.editManifest(parts, ['("case_id": "' golden{1} '"[^}]*?"' field '_sha256": ")[0-9a-f]{64}'], ...
                    ['$1' PackageTools.sha256(bytes)]);
            end
        end

        function parts = editManifest(parts, pattern, replacement)
            k = find(parts.names == "manifest.json", 1);
            parts.bytes{k} = uint8(regexprep(char(parts.bytes{k}(:).'), pattern, replacement)).';
        end

        function parts = editJson(parts, name, pattern, replacement)
            % A text edit of a JSON file of the package with its hash kept right; a changed count of matches is a test bug.
            text = FixedTools.textOf(parts, name);
            edited = regexprep(text, pattern, replacement);
            assert(~strcmp(text, edited), 'The edit of %s changed nothing: %s', name, pattern);
            parts = FixedTools.replaceBytes(parts, name, uint8(edited).');
        end

        function parts = editSpec(parts, pattern, replacement)
            % The same text edit in spec.json and in the specification the manifest records (they must agree).
            parts = FixedTools.editJson(parts, "spec.json", pattern, replacement);
            parts = FixedTools.editManifest(parts, pattern, replacement);
        end

        function parts = editGolden(parts, id, file, positions, change)
            % Change int16 values (1-based POSITIONS in the file's flat order) of a golden file, hash kept right.
            name = "golden/" + id + "/" + file;
            k = find(parts.names == name, 1);
            values = double(typecast(parts.bytes{k}, 'int16'));
            values(positions) = values(positions) + change;
            parts = FixedTools.replaceBytes(parts, name, typecast(int16(values), 'uint8'));
        end

        function f = randomFormat(stream, hot)
            % A random but valid set of formats and weights, in the form opendpd.runtime.fixedGruRun takes. HOT makes the
            % sums saturate: a pre-activation that holds only +-2, biases anywhere in it, weights near their limit, and
            % tables that reach beyond +-2, so that saturating a sum before the table lookup changes the result.
            if nargin < 2
                hot = false;
            end
            integer = @(low, high) low + floor(rand(stream) * (high - low + 1));
            word = @(bits) struct('bits', bits, 'min', -2^(bits - 1), 'max', 2^(bits - 1) - 1);
            x = word(integer(8, 16)); h = word(integer(10, 16)); y = word(integer(8, 16));
            pre = word(integer(24, 32));
            hidden = integer(1, 7);
            weightBits = integer(8, 16);
            wLimit = 2^(weightBits - 1);
            fractions = struct('x', integer(4, 14), 'h', integer(8, 15), 'y', integer(4, 14), 'pre', integer(12, 22), ...
                'ih', integer(5, 14), 'hh', integer(5, 14), 'out', integer(5, 14));
            weightScale = 0.6;
            biasScale = 2^(pre.bits - 3) / pre.max;
            sigmoidRange = integer(2, 8);
            tanhRange = integer(2, 4);
            if hot
                pre = word(24);
                fractions.pre = 22;
                weightScale = 0.95;
                biasScale = 1;
                sigmoidRange = integer(4, 8);
                tanhRange = integer(3, 4);
            end
            weights = @(rows, columns) round((rand(stream, rows, columns) * 2 - 1) * wLimit * weightScale);
            biases = @(rows) round((rand(stream, rows, 1) * 2 - 1) * pre.max * biasScale);
            tableFor = @(range, indexFrac, valueBits) struct( ...
                'values', round((rand(stream, 2 * range * 2^indexFrac, 1) * 2 - 1) * (2^(valueBits - 1) - 1)), ...
                'range', range * 2^indexFrac, 'indexFrac', indexFrac);
            f = struct('hidden', hidden, 'outputs', 2, 'accBits', 48, 'accLimit', 2^47, ...
                'xBits', x.bits, 'xFrac', fractions.x, 'xMin', x.min, 'xMax', x.max, ...
                'hBits', h.bits, 'hFrac', fractions.h, 'hMin', h.min, 'hMax', h.max, ...
                'yBits', y.bits, 'yFrac', fractions.y, 'yMin', y.min, 'yMax', y.max, ...
                'preBits', pre.bits, 'preFrac', fractions.pre, 'preMin', pre.min, 'preMax', pre.max, ...
                'fIH', fractions.ih, 'fHH', fractions.hh, 'fOut', fractions.out, ...
                'wIH', weights(3 * hidden, 2), 'wHH', weights(3 * hidden, hidden), 'wOut', weights(2, hidden), ...
                'bIH', biases(3 * hidden), 'bHH', biases(3 * hidden), 'bOut', biases(2), ...
                'sigmoid', tableFor(sigmoidRange, integer(2, 8), 16), 'tanh', tableFor(tanhRange, integer(2, 8), 16));
        end

        function [y, h, trace] = referenceRun(f, x, h, resetAt)
            % The specification as scalar int64 code, written after the C99 reference (loops, one multiply-add at a time,
            % floor division by idivide), not after the vectorised double code it checks.
            n = size(x, 1);
            hidden = f.hidden;
            x = int64(x);
            h = int64(h);
            y = zeros(n, 2);
            trace = zeros(n, hidden);
            for k = 1:n
                if any(resetAt == k)
                    h(:) = 0;
                end
                xk = FixedTools.saturate(x(k, :), f.xMin, f.xMax);
                ai = zeros(3 * hidden, 1, 'int64');
                ah = zeros(3 * hidden, 1, 'int64');
                for g = 1:3 * hidden
                    accI = int64(0);
                    accH = int64(0);
                    for j = 1:2
                        accI = accI + int64(f.wIH(g, j)) * xk(j);
                    end
                    for j = 1:hidden
                        accH = accH + int64(f.wHH(g, j)) * h(j);
                    end
                    ai(g) = FixedTools.saturate(FixedTools.rescale(accI, f.fIH + f.xFrac, f.preFrac) + int64(f.bIH(g)), ...
                        f.preMin, f.preMax);
                    ah(g) = FixedTools.saturate(FixedTools.rescale(accH, f.fHH + f.hFrac, f.preFrac) + int64(f.bHH(g)), ...
                        f.preMin, f.preMax);
                end
                hNew = zeros(hidden, 1, 'int64');
                for i = 1:hidden
                    r = FixedTools.lookup(FixedTools.saturate(ai(i) + ah(i), f.preMin, f.preMax), f.sigmoid, f.preFrac);
                    z = FixedTools.lookup(FixedTools.saturate(ai(hidden + i) + ah(hidden + i), f.preMin, f.preMax), ...
                        f.sigmoid, f.preFrac);
                    t = FixedTools.rescale(r * ah(2 * hidden + i), f.hFrac + f.preFrac, f.preFrac);
                    c = FixedTools.lookup(FixedTools.saturate(ai(2 * hidden + i) + t, f.preMin, f.preMax), f.tanh, f.preFrac);
                    one = bitshift(int64(1), f.hFrac);
                    mix = (one - z) * c + z * h(i);
                    hNew(i) = FixedTools.saturate(FixedTools.rescale(mix, 2 * f.hFrac, f.hFrac), f.hMin, f.hMax);
                end
                for o = 1:2
                    acc = int64(0);
                    for j = 1:hidden
                        acc = acc + int64(f.wOut(o, j)) * hNew(j);
                    end
                    pre = FixedTools.saturate(FixedTools.rescale(acc, f.fOut + f.hFrac, f.preFrac) + int64(f.bOut(o)), ...
                        f.preMin, f.preMax);
                    y(k, o) = double(FixedTools.saturate(FixedTools.rescale(pre, f.preFrac, f.yFrac), f.yMin, f.yMax));
                end
                h = hNew;
                trace(k, :) = double(h).';
            end
            h = double(h);
        end

        function v = saturate(v, low, high)
            v = min(max(v, int64(low)), int64(high));
        end

        function v = rescale(v, fromFrac, toFrac)
            if toFrac >= fromFrac
                v = v * bitshift(int64(1), toFrac - fromFrac);
            else
                s = fromFrac - toFrac;
                v = idivide(v + bitshift(int64(1), s - 1), bitshift(int64(1), s), 'floor');
            end
        end

        function v = lookup(pre, table, preFrac)
            index = FixedTools.rescale(pre, preFrac, table.indexFrac);
            index = FixedTools.saturate(index, -table.range, table.range - 1) + int64(table.range);
            v = int64(table.values(double(index) + 1));
        end
    end
end
