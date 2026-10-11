function hash = sha256Data(bytes)
%SHA256DATA Lowercase hex SHA-256 of a uint8 vector.
arguments
    bytes (:,1) uint8
end
digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(bytes);
hash = lower(reshape(dec2hex(typecast(int8(digest.digest()), 'uint8'), 2).', 1, []));
end
