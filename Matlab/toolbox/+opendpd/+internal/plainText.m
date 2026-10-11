function value = plainText(value)
%PLAINTEXT Replace the control characters of every string inside VALUE (strings, structs and cells, as jsondecode makes them).
%   Text from a package ends up in messages and in the display of a model. A terminal can act on escape sequences, a carriage
%   return rewrites a line and a bell makes noise, so the characters below space (except tab and line feed), delete and the
%   C1 controls become "?". Everything else, including every other character of every language, is kept.
if ischar(value)
    value(value < 32 & value ~= 9 & value ~= 10 | value == 127 | (value >= 128 & value < 160)) = '?';
elseif isstruct(value)
    names = fieldnames(value);
    for k = 1:numel(value)
        for n = 1:numel(names)
            value(k).(names{n}) = opendpd.internal.plainText(value(k).(names{n}));
        end
    end
elseif iscell(value)
    for k = 1:numel(value)
        value{k} = opendpd.internal.plainText(value{k});
    end
end
end
