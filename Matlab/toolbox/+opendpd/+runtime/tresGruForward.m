function y = tresGruForward(x, w)
%TRESGRUFORWARD The tres_gru backbone on one segment X (complex column), zero state.
% A bias-free GRU stack over the six features of opendpd.runtime.tresFeatures, a linear head (no bias) and the TCN skip
% path of opendpd.runtime.tresSkip added to its output.
iq = [real(x(:)), imag(x(:))];
skip = opendpd.runtime.tresSkip(iq, w.conv1, w.conv2);
h = opendpd.runtime.tresFeatures(iq);
for l = 1:numel(w.weightIH)
    h = opendpd.runtime.gruLayer(h, w.weightIH{l}, w.weightHH{l}, w.biasIH{l}, w.biasHH{l}, zeros(size(w.weightHH{l}, 2), 1));
end
out = h * w.fcWeight.' + skip;
y = complex(out(:, 1), out(:, 2));
end
