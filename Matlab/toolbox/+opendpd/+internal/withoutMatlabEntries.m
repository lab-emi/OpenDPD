function value = withoutMatlabEntries(value, root, separator)
%WITHOUTMATLABENTRIES Remove the entries of a path list that lie under the MATLAB installation.
% MATLAB puts its own library folders at the front of LD_LIBRARY_PATH (DYLD_* on macOS, PATH on Windows). A Python
% started from MATLAB inherits them, and libraries such as libstdc++ then come from MATLAB instead of the system.
% Entries that are not under ROOT, such as the user's CUDA libraries, are kept. An empty result is returned as ''.
arguments
    value (1,1) string
    root (1,1) string
    separator (1,1) string = string(pathsep)
end
entries = split(value, separator);
entries = entries(strlength(entries) > 0);
under = strcmp(entries, root) | startsWith(entries, root + "/") | startsWith(entries, root + "\");
if ispc                                   % Windows paths do not care about case
    under = strcmpi(entries, root) | startsWith(entries, root + "/", IgnoreCase=true) | ...
        startsWith(entries, root + "\", IgnoreCase=true);
end
value = char(strjoin(entries(~under), separator));
end
