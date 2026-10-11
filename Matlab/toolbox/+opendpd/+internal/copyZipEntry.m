function copyZipEntry(archive, name, target, declared)
%COPYZIPENTRY Copy one entry of a java.util.zip.ZipFile to TARGET, moving at most DECLARED+1 bytes (done on the Java side).
% Any count other than DECLARED means the entry is not what the archive claims, so an archive that declares a small
% entry and inflates to a large one is refused after DECLARED+1 bytes, not after it has filled the disk.
try
    input = archive.getInputStream(archive.getEntry(char(name)));
    closeInput = onCleanup(@() input.close());
    output = java.io.FileOutputStream(char(target));
    closeOutput = onCleanup(@() output.close());
    moved = output.getChannel().transferFrom(java.nio.channels.Channels.newChannel(input), 0, declared + 1);
catch cause
    error('opendpd:PackageContents', 'Entry "%s" could not be read from the archive (%s).', name, cause.message);
end
if moved ~= declared
    error('opendpd:PackageContents', 'Entry "%s" does not have the size it declares (%g bytes declared, at least %g found).', ...
        name, declared, moved);
end
end
