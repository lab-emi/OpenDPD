function y = gmpForward(x, weight, M, D)
%GMPFORWARD The gradient-trained GMP backbone (real weights) on one segment, zero history.
% With xp = [zeros(M-1,1); x] and ap = abs([zeros(M-1,1); xp]) (the amplitude is padded twice, as in
% backbones/gmp.py), output j (0-based) is
%   sum_k w(k) xp(j+k)  +  sum_{i=1..D-1} sum_{m=0..M-1} sum_{k=0..M-1} w(M + (((i-1) M + m) M + k)) xp(j+k) ap(j+m+k)^i
% for k = 0..M-1. In terms of the signal this is x(n-a) |x(n-a-b)|^i with a = M-1-k and b = M-1-m, both 0..M-1.
x = x(:);
n = numel(x);
xp = [zeros(M - 1, 1); x];
ap = abs([zeros(M - 1, 1); xp]);
expected = M * (1 + (D - 1) * M);
if numel(weight) ~= expected
    error('opendpd:Package', 'gmp expects %d weights, the package has %d.', expected, numel(weight));
end
y = complex(zeros(n, 1));
for k = 0:M-1
    y = y + weight(k + 1) * xp(k + (1:n));
end
for i = 1:D-1
    for m = 0:M-1
        for k = 0:M-1
            index = M + ((i - 1) * M + m) * M + k + 1;
            y = y + weight(index) * (xp(k + (1:n)) .* ap(m + k + (1:n)) .^ i);
        end
    end
end
end
