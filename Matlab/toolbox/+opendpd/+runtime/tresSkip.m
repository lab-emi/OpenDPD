function skip = tresSkip(iq, conv1, conv2)
%TRESSKIP The tres_gru skip path on IQ (N-by-2), returned N-by-2.
% conv1d(2->3, kernel 3, dilation 16, zero padding 16) -> hardswish -> conv1d(3->2, kernel 1) -> hardswish, which reads
% x(n-16), x(n), x(n+16) with zeros outside the segment. CONV1 is 3-by-2-by-3 and CONV2 is 2-by-3 (2-by-3-by-1).
n = size(iq, 1);
hidden = zeros(3, n);
for j = 0:2
    shift = 16 * (j - 1);                                   % -16, 0, +16
    source = (1:n) + shift;
    valid = source >= 1 & source <= n;
    shifted = zeros(2, n);
    shifted(:, valid) = iq(source(valid), :).';
    for o = 1:3
        hidden(o, :) = hidden(o, :) + squeeze(conv1(o, :, j + 1)) * shifted;
    end
end
hidden = opendpd.runtime.hardswish(hidden);
skip = opendpd.runtime.hardswish(conv2(:, :, 1) * hidden).';
end
