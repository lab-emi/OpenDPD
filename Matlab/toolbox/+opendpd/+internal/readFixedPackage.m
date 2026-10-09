function package = readFixedPackage(file, options)
%READFIXEDPACKAGE Read a fixed-point-v1 deployment zip as data. Nothing in it is executed or trusted blindly:
%   * only the known file names are accepted (golden/<case>/ with five fixed file names, three C sources, three JSON files and
%     the README), none twice, and no path from the archive is ever used: files are copied to names this function chooses;
%   * an entry is copied out with a hard byte limit that depends on what the file is, so an archive that claims to be small
%     and inflates is refused, and a package cannot ask for more golden samples than opendpd.internal.fixedLimits allows;
%   * every file's SHA-256 must equal the manifest's, the manifest must list exactly the files that are in the archive, and the
%     hashes of the golden vectors that the manifest indexes must be the hashes of those files;
%   * JSON is measured before it is parsed (nesting, element count and the length of any number, opendpd.internal.decodeJson),
%     parsed as JSON, checked member by member (only the members the format defines, of the types it defines) and its text is
%     made plain;
%   * the golden vectors are read as little-endian int16, nothing else.
%   The C sources are checked against their hashes like every other file and are never compiled or run by the toolbox.
arguments
    file (1,1) string {mustBeFile}
    options.MaxEntryBytes (1,1) double {mustBePositive} = 256e6
    options.MaxTotalBytes (1,1) double {mustBePositive} = 512e6
end
limits = opendpd.internal.fixedLimits();
fixedNames = ["manifest.json", "spec.json", "weights.json", "README.md", "c/gru_fixed.c", "c/gru_fixed.h", "c/harness.c"];
% The six golden vectors of the specification are the only cases a package holds, so the case names are fixed here.
caseList = ["normal", "extreme", "saturation", "all_zero", "state_reset", "long_sequence"];
caseNames = char(strjoin(caseList, '|'));
casePattern = ['^golden/(' caseNames ')/(x\.i16|y\.i16|h_final\.i16|h_trace\.i16|meta\.json)$'];
directoryPattern = ['^(c|golden|golden/(' caseNames '))/$'];
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
    limit = min(options.MaxEntryBytes, entryLimit(name, limits));
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
if sum(sizes(endsWith(names, "/x.i16"))) / 4 > limits.MaxGoldenSamples
    error('opendpd:PackageContents', 'The golden vectors hold more than %d samples in all, which is more than this toolbox replays.', ...
        limits.MaxGoldenSamples);
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

manifest = decodeFile(local("manifest.json"), 'manifest.json', limits.ManifestSeparators, limits);
if ~(isstruct(manifest) && isscalar(manifest) && isfield(manifest, 'spec') && isstruct(manifest.spec) && isscalar(manifest.spec) ...
        && isfield(manifest.spec, 'spec_id') && ischar(manifest.spec.spec_id) && strcmp(manifest.spec.spec_id, 'fixed-point-v1'))
    shownId = '?';
    if isstruct(manifest) && isscalar(manifest) && isfield(manifest, 'spec') && isstruct(manifest.spec) ...
            && isscalar(manifest.spec) && isfield(manifest.spec, 'spec_id')
        shownId = shown(manifest.spec.spec_id);
    end
    error('opendpd:Package', 'Not a fixed-point-v1 package (specification "%s").', shownId);
end
checkManifest(manifest, limits, caseList);
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
% The manifest indexes the golden vectors with their hashes: they must be the hashes of the files that were just checked.
hashFields = {'input_sha256', 'x.i16'; 'output_sha256', 'y.i16'; 'state_sha256', 'h_final.i16'; 'trace_sha256', 'h_trace.i16'};
for k = 1:numel(manifest.golden)
    for field = hashFields.'
        key = matlab.lang.makeValidName(sprintf('golden/%s/%s', manifest.golden(k).case_id, field{2}));
        if ~isfield(manifest.files, key)
            error('opendpd:PackageContents', 'The archive has no %s for golden case %s.', field{2}, manifest.golden(k).case_id);
        end
        if ~strcmp(manifest.golden(k).(field{1}), manifest.files.(key))
            error('opendpd:PackageHash', 'The manifest''s %s of golden case %s is not the hash of its %s.', field{1}, ...
                manifest.golden(k).case_id, field{2});
        end
    end
end

spec = decodeFile(local("spec.json"), 'spec.json', limits.SpecSeparators, limits);
if ~isequaln(spec, manifest.spec)
    error('opendpd:Package', 'spec.json differs from the specification recorded in the manifest.');
end
% The weights file holds exactly the elements the hidden size and the table limit allow, and not one more.
hidden = manifest.hidden_size;
elements = 3 * hidden^2 + 14 * hidden + 2 + 2 * limits.MaxTableEntries;
weights = decodeFile(local("weights.json"), 'weights.json', elements + 1000, limits);
golden = struct();
for k = 1:numel(names)
    parts = regexp(char(names(k)), casePattern, 'tokens', 'once');
    if isempty(parts) || strcmp(parts{2}, 'meta.json')
        continue
    end
    id = parts{1};
    if ~isfield(golden, id)
        golden.(id) = struct();
    end
    golden.(id).(strrep(parts{2}, '.i16', '')) = readInt16(local(names(k)));
end
package = struct('manifest', opendpd.internal.plainText(manifest), 'spec', spec, 'weights', weights, 'golden', golden, ...
    'sha256', opendpd.internal.sha256(file));
end

function limit = entryLimit(name, limits)
% The most a file of this kind can hold for the limits of this toolbox (the sizes OpenDPD writes are far below them).
if name == "manifest.json"
    limit = limits.ManifestBytes;
elseif name == "spec.json"
    limit = limits.SpecBytes;
elseif name == "weights.json"
    limit = limits.WeightsBytes;
elseif name == "README.md" || startsWith(name, "c/")
    limit = limits.SourceBytes;
elseif endsWith(name, "/meta.json")
    limit = limits.MetaBytes;
elseif endsWith(name, "/h_trace.i16")
    limit = limits.TraceBytes;
elseif endsWith(name, "/h_final.i16")
    limit = 2 * limits.MaxHidden;
else
    limit = 4 * limits.MaxGoldenSamples;                 % x.i16 and y.i16: two int16 per sample
end
end

% ----- the manifest ---------------------------------------------------------------------------------------------------------
% The shape and the types of everything the toolbox reads or shows; the ranges of the numbers it computes with are checked where
% they are used (opendpd.FixedModel). The members are the ones opendpd/schemas/fixed_point.py defines and no others (the Python
% schema forbids extras too), so nothing unchecked rides along in a package.

function checkManifest(m, limits, cases)
opendpd.internal.requireMembers(m, ["schema_version", "spec", "run_id", "model_key", "weights_sha256", "hidden_size", "tensors", "golden", ...
    "verification", "report", "files", "software", "created_at"], "weights_sha256", 'The manifest');
whole(m.schema_version, 1, 1, 'schema_version');
textMatching(m.run_id, '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$', 'run_id');
textMatching(m.model_key, '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$', 'model_key');
if isfield(m, 'weights_sha256') && ~isempty(m.weights_sha256)
    textMatching(m.weights_sha256, '^[0-9a-f]{64}$', 'weights_sha256');
end
whole(m.hidden_size, 1, limits.MaxHidden, 'hidden_size');
if ~(isstruct(m.files) && isscalar(m.files))
    error('opendpd:Package', 'The manifest''s files must be an object that maps every file to its SHA-256.');
end
for part = ["spec", "report", "software"]
    if ~(isstruct(m.(part)) && isscalar(m.(part)))
        error('opendpd:Package', 'The manifest''s %s must be an object.', part);
    end
end
textMatching(m.created_at, '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+(Z|[+-][0-9]{2}:[0-9]{2})$', 'created_at');

hidden = m.hidden_size;
expectedShape = struct('w_ih', [3 * hidden, 2], 'w_hh', [3 * hidden, hidden], 'w_out', [2, hidden], 'b_ih', 3 * hidden, ...
    'b_hh', 3 * hidden, 'b_out', 2);
tensors = m.tensors;
if ~(isstruct(tensors) && numel(tensors) == 6)
    error('opendpd:Package', 'The manifest must describe the six quantised tensors.');
end
seen = strings(1, 0);
for k = 1:numel(tensors)
    t = tensors(k);
    opendpd.internal.requireMembers(t, ["name", "shape", "bits", "frac", "max_abs_float", "saturated"], strings(1, 0), 'A tensor of the manifest');
    if ~(ischar(t.name) && isfield(expectedShape, t.name)) || any(seen == string(t.name))
        error('opendpd:Package', 'The manifest lists a tensor that is unknown or listed twice.');
    end
    seen(end+1) = string(t.name); %#ok<AGROW>
    if ~(isnumeric(t.shape) && isequal(double(t.shape(:)), expectedShape.(t.name)(:)))
        error('opendpd:Package', 'The manifest gives the tensor %s a shape other than the one the hidden size implies.', t.name);
    end
    whole(t.bits, 2, 64, 'tensors[].bits');
    whole(t.frac, 0, 62, 'tensors[].frac');
    number(t.max_abs_float, 'tensors[].max_abs_float');
    whole(t.saturated, 0, Inf, 'tensors[].saturated');
end

golden = m.golden;
if ~(isstruct(golden) && numel(golden) == numel(cases))
    error('opendpd:Package', 'The manifest must index exactly the six golden vectors: %s.', strjoin(cases, ', '));
end
ids = strings(1, 0);
for k = 1:numel(golden)
    g = golden(k);
    opendpd.internal.requireMembers(g, ["case_id", "description", "n_samples", "resets_at", "input_sha256", "output_sha256", "state_sha256", ...
        "trace_sha256"], strings(1, 0), 'A golden case of the manifest');
    if ~(ischar(g.case_id) && any(strcmp(g.case_id, cases)))
        error('opendpd:Package', 'The manifest indexes a golden case that is not one of %s.', strjoin(cases, ', '));
    end
    ids(end+1) = string(g.case_id); %#ok<AGROW>
    textMatching(g.description, '^.{0,400}$', 'golden[].description');
    whole(g.n_samples, 1, limits.MaxGoldenSamples, 'golden[].n_samples');
    if ~(isnumeric(g.resets_at) && isreal(g.resets_at))
        error('opendpd:Package', 'The manifest''s golden[].resets_at must be a list of numbers.');
    end
    for field = ["input_sha256", "output_sha256", "state_sha256", "trace_sha256"]
        textMatching(g.(field), '^[0-9a-f]{64}$', ['golden[].' char(field)]);
    end
end
if numel(unique(ids)) ~= numel(cases)
    error('opendpd:Package', 'The manifest must index each of the six golden vectors once.');
end

v = m.verification;
opendpd.internal.requireMembers(v, ["backend", "status", "compiler", "cases_checked", "mismatch_case", "mismatch_step", "mismatch_signal", ...
    "detail"], strings(1, 0), 'The verification record');
textMatching(v.backend, '^[A-Za-z0-9._ -]{1,40}$', 'verification.backend');
if ~(ischar(v.status) && any(strcmp(v.status, {'bit_exact', 'mismatch', 'not_run'})))
    error('opendpd:Package', 'The manifest''s verification status is not one of bit_exact, mismatch, not_run.');
end
whole(v.cases_checked, 0, 6, 'verification.cases_checked');
for field = ["compiler", "mismatch_case", "mismatch_signal", "detail"]
    if ~(isempty(v.(field)) || (ischar(v.(field)) && size(v.(field), 1) == 1 && numel(v.(field)) <= 2000))
        error('opendpd:Package', 'The manifest''s verification.%s must be text or null.', field);
    end
end
if ~(isempty(v.mismatch_step) || (isnumeric(v.mismatch_step) && isscalar(v.mismatch_step) && isreal(v.mismatch_step)))
    error('opendpd:Package', 'The manifest''s verification.mismatch_step must be a number or null.');
end
end

function whole(value, low, high, label)
if ~(isnumeric(value) && isscalar(value) && isreal(value) && isfinite(value) && value == fix(value) ...
        && value >= low && value <= high)
    error('opendpd:Package', 'The manifest''s %s must be a whole number from %g to %g.', label, low, high);
end
end

function number(value, label)
if ~(isnumeric(value) && isscalar(value) && isreal(value) && isfinite(value))
    error('opendpd:Package', 'The manifest''s %s must be a finite number.', label);
end
end

function textMatching(value, pattern, label)
if ~(ischar(value) && (isempty(value) || size(value, 1) == 1) && ~isempty(regexp(value, pattern, 'once')))
    error('opendpd:Package', 'The manifest''s %s is missing or has an unexpected form.', label);
end
end

function value = decodeFile(path, label, separators, limits)
value = opendpd.internal.decodeJson(readBytes(path), label, MaxDepth=limits.MaxNesting, MaxSeparators=separators);
end

function bytes = readBytes(path)
fid = fopen(path, 'r');
if fid < 0
    error('opendpd:Package', 'Cannot read %s.', path);
end
closeFile = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');
end

function values = readInt16(path)
info = dir(path);
if mod(info.bytes, 2) ~= 0
    error('opendpd:Package', 'A golden vector file does not hold whole 16-bit values (%d bytes).', info.bytes);
end
fid = fopen(path, 'r');
if fid < 0
    error('opendpd:Package', 'Cannot read %s.', path);
end
closeFile = onCleanup(@() fclose(fid));
values = fread(fid, Inf, 'int16=>int16', 0, 'ieee-le');
end

function text = shown(value)
% A value from the package, made safe to put in an error message.
if ischar(value) || isstring(value)
    text = regexprep(char(string(value)), '[^A-Za-z0-9_. -]', '?');
    text = text(1:min(numel(text), 60));
elseif isnumeric(value) && isscalar(value) && isreal(value)
    text = sprintf('%g', value);
else
    text = '?';
end
end
