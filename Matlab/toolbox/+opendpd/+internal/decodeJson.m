function value = decodeJson(bytes, label, options)
%DECODEJSON Parse JSON text that comes from an archive, after checking that its shape cannot hurt MATLAB.
%   VALUE = decodeJson(BYTES, LABEL) measures the text first (opendpd.internal.jsonShape) and refuses it with
%   opendpd:Package if it nests deeper than MaxDepth levels, holds more than MaxSeparators commas outside strings, or has a
%   number or word longer than MaxToken characters: jsondecode recurses per level of nesting, so a very deep text crashes
%   MATLAB; it needs a few hundred bytes per array element, so a file of a few megabytes can ask for gigabytes; and a number
%   with a million digits after the decimal point takes it half a minute. A file that is valid JSON of a sane shape is parsed
%   by jsondecode; anything jsondecode rejects is reported as opendpd:Package, with jsondecode's own message (which can quote
%   the text of the file in full) cut short and made plain. LABEL names the file in messages and is chosen by the caller, never
%   taken from the archive. The measuring is done on the text as jsondecode will see it, after the UTF-8 decoding.
arguments
    bytes (:,1) uint8
    label (1,1) string
    options.MaxDepth (1,1) double = 8
    options.MaxSeparators (1,1) double = 5000
    options.MaxToken (1,1) double = 64
    options.ChunkBytes (1,1) double = 2^20
end
text = native2unicode(bytes.', 'UTF-8');
[depth, separators, token] = opendpd.internal.jsonShape(uint8(text(:)), options.ChunkBytes);
if depth > options.MaxDepth
    error('opendpd:Package', '%s nests %d levels deep; the package files of this format never need more than %d.', ...
        label, depth, options.MaxDepth);
end
if separators > options.MaxSeparators
    error('opendpd:Package', '%s holds %d array elements and object members; the package files of this format never need more than %d.', ...
        label, separators, options.MaxSeparators);
end
if token > options.MaxToken
    error('opendpd:Package', '%s holds a number or word of %d characters; the package files of this format never need more than %d.', ...
        label, token, options.MaxToken);
end
try
    value = jsondecode(text);
catch cause
    error('opendpd:Package', '%s is not valid JSON (%s).', label, opendpd.internal.plainText(cause.message(1:min(end, 120))));
end
end
