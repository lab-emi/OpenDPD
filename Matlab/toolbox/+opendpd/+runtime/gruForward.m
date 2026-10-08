function [y, state] = gruForward(x, w, state)
%GRUFORWARD The gru backbone (nn.GRU stack + linear head) on I/Q samples X (complex column); W is the model's weights.
% STATE is the H-by-L hidden state to start from (zeros at the start of a segment or stream); returns the outputs
% (complex column) and the state after the last sample.
[h, state] = opendpd.runtime.gruStack([real(x(:)), imag(x(:))], w.weightIH, w.weightHH, w.biasIH, w.biasHH, state);
out = h * w.fcWeight.' + w.fcBias.';
y = complex(out(:, 1), out(:, 2));
end
