function writeNpy(file, array)
%WRITENPY Write a real single/double array as a NumPy .npy file (format 1.0, little-endian, C order).
% The Python side of the process transport reads it with numpy.load(allow_pickle=False). N-by-2 I/Q matrices are the
% intended use; any real array works, and a vector is stored as a one-dimensional array like NumPy would.
arguments
    file (1,1) string
    array {mustBeNumeric, mustBeReal, mustBeFinite}
end
if ~(isa(array, 'single') || isa(array, 'double'))
    error('opendpd:InvalidIQ', 'Only single and double arrays can be written (got %s).', class(array));
end
if isa(array, 'single')
    descr = '<f4';
else
    descr = '<f8';
end
shape = size(array);
if isvector(array) && ~isscalar(array)
    shape = numel(array);
elseif isscalar(array)
    shape = [];
end
if numel(shape) > 1
    data = permute(array, ndims(array):-1:1);          % NumPy's C order: the last index runs fastest
else
    data = array;
end
digits = arrayfun(@(d) sprintf('%d', d), shape, UniformOutput=false);
switch numel(shape)
    case 0, shapeText = '()';
    case 1, shapeText = ['(' digits{1} ',)'];
    otherwise, shapeText = ['(' strjoin(digits, ', ') ')'];
end
header = sprintf('{''descr'': ''%s'', ''fortran_order'': False, ''shape'': %s, }', descr, shapeText);
padding = mod(-(10 + numel(header) + 1), 64);
header = [header repmat(' ', 1, padding) newline];
fid = fopen(file, 'w');
if fid < 0
    error('opendpd:Package', 'Cannot write %s.', file);
end
closeFile = onCleanup(@() fclose(fid));
fwrite(fid, uint8([147 78 85 77 80 89 1 0]), 'uint8');
fwrite(fid, uint16(numel(header)), 'uint16', 0, 'ieee-le');
fwrite(fid, uint8(header), 'uint8');
fwrite(fid, data(:), class(array), 0, 'ieee-le');
end
