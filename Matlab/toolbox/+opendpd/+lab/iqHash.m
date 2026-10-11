function hash = iqHash(z)
%IQHASH SHA-256 of a complex signal as OpenDPD's session records hash it: float32 interleaved I/Q, little-endian.
% The same bytes as numpy's float32 array of shape (n, 2) in C order, so a record made here can be checked in Python.
z = z(:);
iq = single([real(z).'; imag(z).']);
hash = opendpd.internal.sha256Data(typecast(iq(:), 'uint8'));
end
