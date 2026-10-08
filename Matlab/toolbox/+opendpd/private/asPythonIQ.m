function value = asPythonIQ(x)
%ASPYTHONIQ Normalize MATLAB vector orientation and preserve source precision.
try
    validateattributes(x, {'single', 'double'}, {'vector', 'nonempty', 'finite', 'nonsparse'});
catch cause
    error('opendpd:InvalidIQ', 'Expected a finite single/double I/Q vector: %s', cause.message);
end
value = py.numpy.asarray([real(x(:)), imag(x(:))]);
value = value.reshape(int64(numel(x)), int64(2));
end
