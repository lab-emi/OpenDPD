function hash = sha256(file)
%SHA256 Lowercase hex SHA-256 of a file, read in blocks (uses the JVM that MATLAB ships with).
fid = fopen(char(file), 'r');
if fid < 0
    error('opendpd:Package', 'Cannot read %s.', file);
end
closeFile = onCleanup(@() fclose(fid));
digest = java.security.MessageDigest.getInstance('SHA-256');
while true
    block = fread(fid, 1e7, '*uint8');
    if isempty(block)
        break
    end
    digest.update(block);
end
raw = typecast(int8(digest.digest()), 'uint8');
hash = lower(reshape(dec2hex(raw, 2).', 1, []));
end
