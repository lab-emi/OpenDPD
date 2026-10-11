function requireMembers(s, allowed, optional, label)
%REQUIREMEMBERS S must be one object (a scalar struct from jsondecode) with the members ALLOWED and no others.
%   Every name in ALLOWED must be present except those also in OPTIONAL. The names come from the package, so they are
%   shown only after being made plain. Errors are opendpd:Package; LABEL is chosen by the caller.
if ~(isstruct(s) && isscalar(s))
    error('opendpd:Package', '%s must be an object.', label);
end
present = string(fieldnames(s)).';
extra = setdiff(present, string(allowed));
if ~isempty(extra)
    error('opendpd:Package', '%s has a member "%s" that fixed-point-v1 does not define.', label, ...
        regexprep(extra(1).extractBefore(min(strlength(extra(1)), 60) + 1), '[^A-Za-z0-9_. -]', '?'));
end
missing = setdiff(setdiff(string(allowed), string(optional)), present);
if ~isempty(missing)
    error('opendpd:Package', '%s lacks "%s".', label, missing(1));
end
end
