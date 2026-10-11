function [maxDepth, separators, longestToken] = jsonShape(bytes, chunkBytes)
%JSONSHAPE Nesting depth, number of separators and longest bare token of JSON text, measured without parsing it.
%   [DEPTH, COMMAS, TOKEN] = jsonShape(BYTES) scans the UTF-8 bytes of a JSON text (or its character codes, cut to 255) and
%   returns the deepest nesting of brackets and braces, the number of commas outside strings (every array element and object
%   member after the first needs one) and the length of the longest token outside strings: a run of characters that are
%   neither white space, quotes nor structure, which in JSON is a number or one of true, false and null.
%   MATLAB's jsondecode recurses once per level, so a few tens of thousands of levels end the MATLAB process with a
%   segmentation fault; it holds a few hundred bytes per array element, so a small archive can inflate into gigabytes; and its
%   time grows with the square of the number of digits after the decimal point of one number (a million digits take half a
%   minute, thirty million would take hours). Text that comes from an archive is therefore measured first (see
%   opendpd.internal.decodeJson).
%   Strings are skipped: a bracket or comma inside one counts for nothing, and inside a string a backslash escapes the next
%   character, so a quote ends the string only after an even number of backslashes. The text is scanned in chunks of
%   CHUNKBYTES (default 2^20) with the state carried across their boundaries, which keeps the memory use small; the result
%   does not depend on the chunk size. Text that is not valid JSON gets numbers that are exact up to its first error, which
%   is where jsondecode stops.
arguments
    bytes (:,1) uint8
    chunkBytes (1,1) double {mustBeInteger, mustBePositive} = 2^20
end
maxDepth = 0;
separators = 0;
longestToken = 0;
depth = 0;
inString = false;                  % whether the previous chunk ended inside a string
carry = 0;                         % parity of the run of backslashes that ends the previous chunk
tokenRun = 0;                      % length of the token that ends the previous chunk
count = numel(bytes);
for first = 1:chunkBytes:count
    c = bytes(first:min(first + chunkBytes - 1, count));
    positions = (1:numel(c)).';
    isBackslash = c == 92;
    lastOther = cummax(positions .* ~isBackslash);           % the last position at or before each that is not a backslash
    backslashes = positions - lastOther;                     % length of the run of backslashes that ends at each position
    backslashes(lastOther == 0) = backslashes(lastOther == 0) + carry;   % the run that opens the chunk continues the previous chunk's
    parity = mod(backslashes, 2);
    escaped = [carry; parity(1:end-1)] == 1;                 % preceded by an odd run of backslashes
    quotes = c == 34 & ~escaped;
    inside = mod(double(inString) + cumsum(double(quotes)), 2) == 1;
    outside = ~inside;
    level = depth + cumsum(double(outside & (c == 91 | c == 123)) - double(outside & (c == 93 | c == 125)));
    maxDepth = max(maxDepth, max(level));
    separators = separators + nnz(outside & c == 44);
    bare = outside & ~(c == 32 | c == 9 | c == 10 | c == 13 | c == 34 | c == 44 | c == 58 | c == 91 | c == 93 | c == 123 | c == 125);
    lastBreak = cummax(positions .* ~bare);                  % the last position at or before each that is not part of a token
    tokenLength = positions - lastBreak;                     % length of the token that ends at each position
    tokenLength(lastBreak == 0) = tokenLength(lastBreak == 0) + tokenRun;
    longestToken = max(longestToken, max(tokenLength));
    tokenRun = tokenLength(end);
    depth = level(end);
    inString = inside(end);
    carry = parity(end);
end
end
