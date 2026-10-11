function value = asPythonIQ(x)
%ASPYTHONIQ Normalize MATLAB vector orientation and preserve source precision.
% Shared by the toolbox's functions and its sub-packages (a private folder is invisible to +metrics).
try
    validateattributes(x, {'single', 'double'}, {'vector', 'nonempty', 'finite', 'nonsparse'});
catch cause
    error('opendpd:InvalidIQ', 'Expected a finite single/double I/Q vector: %s', cause.message);
end
value = py.numpy.asarray([real(x(:)), imag(x(:))]);
value = value.reshape(int64(numel(x)), int64(2));
end
