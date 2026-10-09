function [y, h, trace] = fixedGruRun(x, h, resetAt, f)
%FIXEDGRURUN The GRU of OpenDPD's fixed-point-v1 specification, integer arithmetic carried exactly in double precision.
%   [y, h, trace] = opendpd.runtime.fixedGruRun(x, h, resetAt, f) runs the rows of X (integers in the input format, one
%   sample per row) from the state H (a hidden-by-1 column of integers in the state format). It returns Y (one row of
%   output integers per sample), the final state, and TRACE, the state after every sample (one row per sample).
%   Before sample k with k in RESETAT (1-based; entries outside 1..N change nothing) the state is zeroed. F carries the formats
%   and the integer weights; opendpd.FixedModel builds it from a package and has checked that no value below can reach 2^53,
%   so that every sum, product and shift is exact.
%   The steps and their order are those of docs/protocols/fixed-point-v1.md section 3. An accumulator that reaches the
%   declared width raises opendpd:FixedOverflow, as the Python reference does.
n = size(x, 1);
hidden = f.hidden;
y = zeros(n, f.outputs);
trace = zeros(n, hidden);
one = 2^f.hFrac;
isReset = false(n, 1);
isReset(resetAt(resetAt >= 1 & resetAt <= n)) = true;      % a mask: a long list of resets costs nothing per sample
for k = 1:n
    if isReset(k)
        h = zeros(hidden, 1);
    end
    xk = min(max(x(k, :).', f.xMin), f.xMax);
    accI = f.wIH * xk;
    accH = f.wHH * h;
    if any(abs(accI) >= f.accLimit) || any(abs(accH) >= f.accLimit)
        error('opendpd:FixedOverflow', 'An accumulator reaches %d bits at sample %d.', f.accBits, k);
    end
    ai = min(max(opendpd.runtime.fixedRescale(accI, f.fIH + f.xFrac, f.preFrac) + f.bIH, f.preMin), f.preMax);
    ah = min(max(opendpd.runtime.fixedRescale(accH, f.fHH + f.hFrac, f.preFrac) + f.bHH, f.preMin), f.preMax);
    gate = 1:hidden;
    r = lookup(min(max(ai(gate) + ah(gate), f.preMin), f.preMax), f.sigmoid, f.preFrac);
    z = lookup(min(max(ai(hidden + gate) + ah(hidden + gate), f.preMin), f.preMax), f.sigmoid, f.preFrac);
    t = opendpd.runtime.fixedRescale(r .* ah(2 * hidden + gate), f.hFrac + f.preFrac, f.preFrac);
    c = lookup(min(max(ai(2 * hidden + gate) + t, f.preMin), f.preMax), f.tanh, f.preFrac);
    mixed = (one - z) .* c + z .* h;
    hNew = min(max(opendpd.runtime.fixedRescale(mixed, 2 * f.hFrac, f.hFrac), f.hMin), f.hMax);
    accO = f.wOut * hNew;
    if any(abs(accO) >= f.accLimit)
        error('opendpd:FixedOverflow', 'An accumulator reaches %d bits at sample %d.', f.accBits, k);
    end
    pre = min(max(opendpd.runtime.fixedRescale(accO, f.fOut + f.hFrac, f.preFrac) + f.bOut, f.preMin), f.preMax);
    y(k, :) = min(max(opendpd.runtime.fixedRescale(pre, f.preFrac, f.yFrac), f.yMin), f.yMax).';
    h = hNew;
    trace(k, :) = h.';
end
end

function value = lookup(pre, table, preFrac)
% The table entry for a pre-activation: round it to the table's index fraction, clamp it to the range, add the offset.
index = opendpd.runtime.fixedRescale(pre, preFrac, table.indexFrac);
index = min(max(index, -table.range), table.range - 1) + table.range;
value = table.values(index + 1);
end
