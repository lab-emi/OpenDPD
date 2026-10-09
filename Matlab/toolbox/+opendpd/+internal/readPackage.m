function package = readPackage(file, options)
%READPACKAGE Read an opendpd-model-v1 zip as data. Nothing in it is executed or trusted blindly:
%   * only the known file names are accepted (no path from the archive is used), none twice;
%   * an entry is copied out with a hard byte limit, so an archive that claims to be small and inflates is refused;
%   * every file's SHA-256 must equal the manifest's;
%   * the manifest is measured before it is parsed (nesting, element count and the length of any number,
%     opendpd.internal.decodeJson) and the control characters of its text are replaced, because jsondecode recurses per level,
%     takes quadratic time on a long number and a terminal acts on escape sequences;
%   * the arrays come from the .npz files through a strict parser (readNpz/readNpy) that only ever reads bytes as numbers.
%     The .mat files are hash-checked but never opened: listing a MAT file (whos -file) and loading it both call loadobj
%     for classes on the MATLAB path, so inspecting a MAT file from an untrusted source is itself a way to run code.
arguments
    file (1,1) string {mustBeFile}
    options.MaxEntryBytes (1,1) double {mustBePositive} = 256e6
    options.MaxTotalBytes (1,1) double {mustBePositive} = 512e6
end
allowed = ["manifest.json", "README.md", "weights.mat", "weights.npz", "golden/golden.mat", "golden/golden.npz"];
try
    archive = java.util.zip.ZipFile(char(file));
catch cause
    error('opendpd:Package', '%s is not a readable zip file (%s).', file, cause.message);
end
closeArchive = onCleanup(@() archive.close());

names = strings(0, 1);
sizes = zeros(0, 1);
entries = archive.entries();
while entries.hasMoreElements()
    entry = entries.nextElement();
    name = string(entry.getName());
    if name == "golden/" && entry.isDirectory()
        continue
    end
    if ~any(name == allowed)
        error('opendpd:PackageContents', ['Unexpected entry "%s". An opendpd-model-v1 package holds only %s; nothing else ' ...
            'is read or extracted.'], name, strjoin(allowed, ', '));
    end
    if any(names == name)
        error('opendpd:PackageContents', 'Entry "%s" appears more than once.', name);
    end
    size = double(entry.getSize());
    limit = options.MaxEntryBytes;
    if name == "manifest.json"
        limit = min(limit, 4e6);                       % OpenDPD writes a few kilobytes
    end
    if size < 0 || size > limit
        error('opendpd:PackageContents', 'Entry "%s" has an unusable size (%g bytes; the limit for this file is %g).', name, ...
            size, limit);
    end
    names(end+1, 1) = name; %#ok<AGROW>
    sizes(end+1, 1) = size; %#ok<AGROW>
end
if sum(sizes) > options.MaxTotalBytes
    error('opendpd:PackageContents', 'The package claims %g bytes in total; the limit is %g.', sum(sizes), options.MaxTotalBytes);
end
if ~all(ismember(["manifest.json", "weights.npz"], names))
    error('opendpd:PackageContents', 'The package needs manifest.json and weights.npz.');
end

folder = string(tempname);
mkdir(folder);
mkdir(fullfile(folder, 'golden'));
removeFolder = onCleanup(@() rmdir(folder, 's'));
for k = 1:numel(names)
    opendpd.internal.copyZipEntry(archive, names(k), fullfile(folder, strrep(names(k), "/", filesep)), sizes(k));
end

manifest = opendpd.internal.decodeJson(readBytes(fullfile(folder, 'manifest.json')), 'manifest.json', MaxDepth=8, ...
    MaxSeparators=20000);
if ~(isstruct(manifest) && isscalar(manifest) && isfield(manifest, 'format') && ischar(manifest.format) ...
        && strcmp(manifest.format, 'opendpd-model-v1'))
    shown = '?';
    if isstruct(manifest) && isscalar(manifest) && isfield(manifest, 'format') && ischar(manifest.format)
        shown = regexprep(manifest.format(1:min(end, 60)), '[^A-Za-z0-9_. -]', '?');
    end
    error('opendpd:Package', 'Not an opendpd-model-v1 package (format "%s").', shown);
end
% jsondecode turns "golden/golden.mat" into the valid field name "golden_golden_mat".
for name = names.'
    if name == "manifest.json"
        continue
    end
    key = matlab.lang.makeValidName(char(name));
    if ~isfield(manifest, 'files') || ~(isstruct(manifest.files) && isscalar(manifest.files)) || ~isfield(manifest.files, key)
        error('opendpd:PackageHash', 'The manifest does not list %s, so it cannot be verified.', name);
    end
    actual = opendpd.internal.sha256(fullfile(folder, strrep(name, "/", filesep)));
    if ~strcmp(actual, manifest.files.(key))
        error('opendpd:PackageHash', '%s does not match the SHA-256 in the manifest; the package is damaged or was modified.', name);
    end
end

if ~any(names == "weights.npz")
    error('opendpd:PackageContents', 'The package needs weights.npz (the toolbox reads the .npz arrays; the .mat files are for your own code).');
end
limits = {'MaxEntryBytes', options.MaxEntryBytes, 'MaxTotalBytes', options.MaxTotalBytes};
package = struct('manifest', opendpd.internal.plainText(manifest), 'weights', opendpd.internal.readNpz(fullfile(folder, 'weights.npz'), limits{:}), ...
    'golden', struct(), 'sha256', opendpd.internal.sha256(file));
if any(names == "golden/golden.npz")
    package.golden = opendpd.internal.readNpz(fullfile(folder, 'golden', 'golden.npz'), limits{:});
end
end

function bytes = readBytes(path)
fid = fopen(path, 'r');
if fid < 0
    error('opendpd:Package', 'Cannot read %s.', path);
end
closeFile = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');
end
