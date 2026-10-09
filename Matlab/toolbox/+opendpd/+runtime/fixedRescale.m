function v = fixedRescale(v, fromFrac, toFrac)
%FIXEDRESCALE Change the fraction of integers held exactly in double precision, by the rule of fixed-point-v1.
%   v = opendpd.runtime.fixedRescale(v, fromFrac, toFrac) is an exact left shift when toFrac >= fromFrac, otherwise the
%   round-half-up right shift floor((v + 2^(s-1)) / 2^s) with s = fromFrac - toFrac. Dividing by a power of two is exact, so
%   the floor is the integer the C99 reference gets from its division-based shift.
if toFrac >= fromFrac
    v = v * 2^(toFrac - fromFrac);
else
    s = fromFrac - toFrac;
    v = floor((v + 2^(s - 1)) / 2^s);
end
end
