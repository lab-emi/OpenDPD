function array = readNpy(bytes, name)
%READNPY Parse a NumPy .npy array from its bytes. Accepts exactly what NumPy writes for plain numeric arrays:
% little-endian float32/float64/complex64/complex128, C or Fortran order, any shape. Anything else (object arrays,
% structured or byte-swapped types, extra header keys, a data length that does not match the shape) is refused, so the
% bytes are only ever interpreted as numbers - never unpickled, never evaluated.
arguments
    bytes (:,1) uint8
    name (1,1) string
end
if numel(bytes) < 12 || ~isequal(bytes(1:6).', uint8([147 78 85 77 80 89]))
    error('opendpd:PackageContents', '%s is not a NumPy .npy array.', name);
end
switch bytes(7)
    case 1
        headerLength = double(typecast(bytes(9:10), 'uint16'));
        start = 10;
    case {2, 3}
        headerLength = double(typecast(bytes(9:12), 'uint32'));
        start = 12;
    otherwise
        error('opendpd:PackageContents', '%s uses .npy format version %d, which is not supported.', name, bytes(7));
end
dataStart = start + headerLength;
if dataStart > numel(bytes) || any(bytes(start+1:dataStart) > 127)
    error('opendpd:PackageContents', '%s has a damaged or non-ASCII .npy header.', name);
end
header = char(bytes(start+1:dataStart).');
pattern = ['^\{\s*''descr''\s*:\s*''([<|=][a-z][0-9]+)''\s*,\s*''fortran_order''\s*:\s*(True|False)\s*,\s*' ...
    '''shape''\s*:\s*\(([0-9,\s]*)\)\s*,?\s*\}\s*$'];
tokens = regexp(header, pattern, 'tokens', 'once');
if isempty(tokens)
    error('opendpd:PackageContents', '%s has a .npy header this toolbox does not read: %s', name, strtrim(header));
end
descr = tokens{1};
fortran = strcmp(tokens{2}, 'True');
dims = str2double(regexp(tokens{3}, '[0-9]+', 'match'));
if any(dims > 2^31)
    error('opendpd:PackageContents', '%s declares an implausible shape (%s).', name, strtrim(tokens{3}));
end
count = prod(dims);                                % prod([]) is 1: a 0-d array holds one value
switch descr
    case '<f4', itemSize = 4; kind = 'single'; isComplex = false;
    case '<f8', itemSize = 8; kind = 'double'; isComplex = false;
    case '<c8', itemSize = 8; kind = 'single'; isComplex = true;
    case '<c16', itemSize = 16; kind = 'double'; isComplex = true;
    otherwise
        error('opendpd:PackageContents', '%s has element type "%s"; only float32, float64, complex64 and complex128 arrays are read.', ...
            name, descr);
end
if numel(bytes) - dataStart ~= count * itemSize
    error('opendpd:PackageContents', '%s holds %d data bytes but its shape (%s) needs %g.', name, numel(bytes) - dataStart, ...
        strtrim(tokens{3}), count * itemSize);
end
values = typecast(bytes(dataStart+1:end), kind);
if isComplex
    values = complex(values(1:2:end), values(2:2:end));
end
switch numel(dims)
    case 0
        array = values;
    case 1
        array = values(:);
    otherwise
        if fortran
            array = reshape(values, dims);
        else
            array = permute(reshape(values, fliplr(dims)), numel(dims):-1:1);   % C order: the last index runs fastest
        end
end
end
