function y = tresGruForward(x, w)
%TRESGRUFORWARD The tres_gru backbone on one segment X (complex column), zero state.
% Features [I Q |x| |x|^3 I(n+1) Q(n+1)] where the "next" sample of the last sample is the first sample of the
% segment (torch.roll); a bias-free GRU stack; a linear head plus the TCN skip path
% conv1d(2->3, kernel 3, dilation 16, zero padding 16) -> hardswish -> conv1d(3->2, kernel 1) -> hardswish,
% which reads x(n-16), x(n), x(n+16) with zeros outside the segment.
iq = [real(x(:)), imag(x(:))];
n = size(iq, 1);
hidden = zeros(3, n);
for j = 0:2
    shift = 16 * (j - 1);                                   % -16, 0, +16
    source = (1:n) + shift;
    valid = source >= 1 & source <= n;
    shifted = zeros(2, n);
    shifted(:, valid) = iq(source(valid), :).';
    for o = 1:3
        hidden(o, :) = hidden(o, :) + squeeze(w.conv1(o, :, j + 1)) * shifted;
    end
end
hidden = opendpd.runtime.hardswish(hidden);
skip = opendpd.runtime.hardswish(w.conv2(:, :, 1) * hidden);
amplitude = sqrt(iq(:, 1) .^ 2 + iq(:, 2) .^ 2);
next = circshift(iq, -1, 1);
features = [iq, amplitude, amplitude .^ 3, next];
zeroState = zeros(size(w.weightHH{1}, 2), numel(w.weightIH));
h = opendpd.runtime.gruStack(features, w.weightIH, w.weightHH, w.biasIH, w.biasHH, zeroState);
out = h * w.fcWeight.' + skip.';
y = complex(out(:, 1), out(:, 2));
end
