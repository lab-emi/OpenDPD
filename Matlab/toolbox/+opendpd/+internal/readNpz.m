function arrays = readNpz(file, options)
%READNPZ Read a NumPy .npz file (a zip of .npy arrays) into a struct of arrays, with the same discipline as readPackage:
% entry names must be plain identifiers ending in .npy, none twice, sizes bounded on the way in and out, and every array
% goes through readNpy, which only ever interprets bytes as numbers.
arguments
    file (1,1) string {mustBeFile}
    options.MaxEntryBytes (1,1) double {mustBePositive} = 256e6
    options.MaxTotalBytes (1,1) double {mustBePositive} = 512e6
    options.MaxEntries (1,1) double {mustBePositive} = 256
end
try
    archive = java.util.zip.ZipFile(char(file));
catch cause
    error('opendpd:PackageContents', '%s is not a readable .npz file (%s).', file, cause.message);
end
closeArchive = onCleanup(@() archive.close());
names = strings(0, 1);
sizes = zeros(0, 1);
entries = archive.entries();
while entries.hasMoreElements()
    entry = entries.nextElement();
    name = string(entry.getName());
    if isempty(regexp(char(name), '^[A-Za-z][A-Za-z0-9_]{0,63}\.npy$', 'once'))
        error('opendpd:PackageContents', 'The .npz holds an entry named "%s"; only arrays named like "weight_1.npy" are read.', name);
    end
    if any(names == name)
        error('opendpd:PackageContents', 'The .npz holds "%s" twice.', name);
    end
    size = double(entry.getSize());
    if size < 0 || size > options.MaxEntryBytes
        error('opendpd:PackageContents', 'Array "%s" has an unusable size (%g bytes).', name, size);
    end
    names(end+1, 1) = name; %#ok<AGROW>
    sizes(end+1, 1) = size; %#ok<AGROW>
    if numel(names) > options.MaxEntries
        error('opendpd:PackageContents', 'The .npz holds more than %d arrays.', options.MaxEntries);
    end
end
if sum(sizes) > options.MaxTotalBytes
    error('opendpd:PackageContents', 'The .npz claims %g bytes in total; the limit is %g.', sum(sizes), options.MaxTotalBytes);
end
folder = string(tempname);
mkdir(folder);
removeFolder = onCleanup(@() rmdir(folder, 's'));
arrays = struct();
for k = 1:numel(names)
    target = fullfile(folder, sprintf('a%d.npy', k));
    opendpd.internal.copyZipEntry(archive, names(k), target, sizes(k));
    fid = fopen(target, 'r');
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    arrays.(extractBefore(names(k), '.npy')) = opendpd.internal.readNpy(bytes, names(k));
end
end
