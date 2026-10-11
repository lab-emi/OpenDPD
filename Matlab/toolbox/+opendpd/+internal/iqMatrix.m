function iq = iqMatrix(x, name)
%IQMATRIX Validate a PA signal and return it as a real N-by-2 [I Q] matrix of the same precision (a real vector is I only).
arguments
    x
    name (1,1) string = "signal"
end
try
    validateattributes(x, {'single', 'double'}, {'vector', 'nonempty', 'finite', 'nonsparse'});
catch cause
    error('opendpd:InvalidIQ', 'Expected %s to be a finite single/double I/Q vector: %s', name, cause.message);
end
iq = [real(x(:)), imag(x(:))];
end
