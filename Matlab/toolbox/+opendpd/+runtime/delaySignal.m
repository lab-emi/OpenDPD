function y = delaySignal(x, d)
%DELAYSIGNAL Shift a column vector by D samples with zero fill: D > 0 reads the past, D < 0 the future.
% Mirrors opendpd.core.polynomial._delay.
n = numel(x);
x = x(:);
if d >= 0
    k = min(d, n);
    y = [zeros(k, 1); x(1:n-k)];
else
    k = min(-d, n);
    y = [x(k+1:n); zeros(k, 1)];
end
end
