function package = readFixedPackage(file, options)
%READFIXEDPACKAGE Read a fixed-point-v1 deployment zip as data. Nothing in it is executed or trusted blindly:
%   * only the known file names are accepted (golden/<case>/ with five fixed file names, three C sources, three JSON files and
%     the README), none twice, and no path from the archive is ever used: files are copied to names this function chooses;
%   * an entry is copied out with a hard byte limit, so an archive that claims to be small and inflates is refused;
%   * every file's SHA-256 must equal the manifest's, and the manifest must list exactly the files that are in the archive;
%   * JSON is parsed as JSON and the golden vectors are read as little-endian int16, nothing else.
%   The C sources are checked against their hashes like every other file and are never compiled or run by the toolbox.
arguments
    file (1,1) string {mustBeFile}
    options.MaxEntryBytes (1,1) double {mustBePositive} = 256e6
    options.MaxTotalBytes (1,1) double {mustBePositive} = 512e6
end
fixedNames = ["manifest.json", "spec.json", "weights.json", "README.md", "c/gru_fixed.c", "c/gru_fixed.h", "c/harness.c"];
casePattern = '^golden/([a-z][a-z0-9_]{0,39})/(x\.i16|y\.i16|h_final\.i16|h_trace\.i16|meta\.json)$';
directoryPattern = '^(c|golden|golden/[a-z][a-z0-9_]{0,39})/$';
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
    plain = isempty(regexp(char(name), '[^A-Za-z0-9_./-]', 'once'));
    if plain && entry.isDirectory() && ~isempty(regexp(char(name), directoryPattern, 'once'))
        continue
    end
    if ~plain || entry.isDirectory() || ~(any(name == fixedNames) || ~isempty(regexp(char(name), casePattern, 'once')))
        error('opendpd:PackageContents', ['Unexpected entry "%s". A fixed-point-v1 package holds only manifest.json, ' ...
            'spec.json, weights.json, README.md, three C sources under c/ and five files per golden case under ' ...
            'golden/<case>/; nothing else is read or extracted.'], regexprep(char(name), '[^A-Za-z0-9_./-]', '?'));
    end
    if any(names == name)
        error('opendpd:PackageContents', 'Entry "%s" appears more than once.', name);
    end
    size = double(entry.getSize());
    if size < 0 || size > options.MaxEntryBytes
        error('opendpd:PackageContents', 'Entry "%s" has an unusable size (%g bytes; the limit is %g).', name, size, ...
            options.MaxEntryBytes);
    end
    names(end+1, 1) = name; %#ok<AGROW>
    sizes(end+1, 1) = size; %#ok<AGROW>
end
if sum(sizes) > options.MaxTotalBytes
    error('opendpd:PackageContents', 'The package claims %g bytes in total; the limit is %g.', sum(sizes), options.MaxTotalBytes);
end
if ~all(ismember(["manifest.json", "spec.json", "weights.json"], names))
    error('opendpd:PackageContents', 'The package needs manifest.json, spec.json and weights.json.');
end

folder = string(tempname);
mkdir(folder);
removeFolder = onCleanup(@() rmdir(folder, 's'));
local = @(name) fullfile(folder, strrep(name, "/", filesep));
for k = 1:numel(names)
    target = local(names(k));
    if ~isfolder(fileparts(target))
        mkdir(fileparts(target));
    end
    opendpd.internal.copyZipEntry(archive, names(k), target, sizes(k));
end

manifest = decodeJson(local("manifest.json"), 'manifest.json');
if ~isstruct(manifest) || ~isfield(manifest, 'spec') || ~isstruct(manifest.spec) || ~isfield(manifest.spec, 'spec_id') ...
        || ~isequal(manifest.spec.spec_id, 'fixed-point-v1') || ~isfield(manifest, 'files') || ~isstruct(manifest.files)
    shown = '?';
    if isstruct(manifest) && isfield(manifest, 'spec') && isstruct(manifest.spec) && isfield(manifest.spec, 'spec_id')
        shown = regexprep(char(string(manifest.spec.spec_id)), '[^A-Za-z0-9_.-]', '?');
    end
    error('opendpd:Package', 'Not a fixed-point-v1 package (specification "%s").', shown);
end
% jsondecode turns "golden/normal/x.i16" into the valid field name "golden_normal_x_i16": the mapping must not merge entries.
others = names(names ~= "manifest.json");
keys = string(matlab.lang.makeValidName(cellstr(others)));
if numel(unique(keys)) ~= numel(keys)
    error('opendpd:PackageContents', 'Two entries have names that the manifest cannot tell apart.');
end
listed = sort(string(fieldnames(manifest.files)));
if ~isequal(listed(:), sort(keys(:)))
    error('opendpd:PackageHash', ['The manifest lists %d files and the archive holds %d besides the manifest; they must be ' ...
        'the same files.'], numel(listed), numel(keys));
end
for k = 1:numel(others)
    actual = opendpd.internal.sha256(local(others(k)));
    expected = manifest.files.(keys(k));
    if ~ischar(expected) || ~strcmp(actual, expected)
        error('opendpd:PackageHash', '%s does not match the SHA-256 in the manifest; the package is damaged or was modified.', ...
            others(k));
    end
end

spec = decodeJson(local("spec.json"), 'spec.json');
if ~isequaln(spec, manifest.spec)
    error('opendpd:Package', 'spec.json differs from the specification recorded in the manifest.');
end
weights = decodeJson(local("weights.json"), 'weights.json');
golden = struct();
for k = 1:numel(names)
    parts = regexp(char(names(k)), casePattern, 'tokens', 'once');
    if isempty(parts) || strcmp(parts{2}, 'meta.json')
        continue
    end
    id = parts{1};
    if ~isvarname(id)
        error('opendpd:PackageContents', 'The golden case name "%s" cannot be used.', id);
    end
    if ~isfield(golden, id)
        golden.(id) = struct();
    end
    golden.(id).(strrep(parts{2}, '.i16', '')) = readInt16(local(names(k)));
end
package = struct('manifest', manifest, 'spec', spec, 'weights', weights, 'golden', golden, ...
    'sha256', opendpd.internal.sha256(file));
end

function value = decodeJson(path, label)
try
    value = jsondecode(fileread(path));
catch cause
    error('opendpd:Package', '%s is not valid JSON (%s).', label, cause.message);
end
end

function values = readInt16(path)
fid = fopen(path, 'r');
if fid < 0
    error('opendpd:Package', 'Cannot read %s.', path);
end
closeFile = onCleanup(@() fclose(fid));
values = fread(fid, Inf, 'int16=>int16', 0, 'ieee-le');
end
