function [output, h] = gruLayer(input, weightIH, weightHH, biasIH, biasHH, h)
%GRULAYER One torch.nn.GRU layer (batch of one, gate order r, z, n) over INPUT (N-by-F).
% weightIH is 3H-by-F, weightHH is 3H-by-H, biasIH and biasHH are 3H-by-1 (zeros when the model has no bias) and H is the
% hidden state to start from (H-by-1). Returns the layer's outputs (N-by-H) and the state after the last sample.
%   r = sigmoid(Wir x + bir + Whr h + bhr)   z = sigmoid(Wiz x + biz + Whz h + bhz)
%   n = tanh(Win x + bin + r .* (Whn h + bhn))   h' = (1 - z) .* n + z .* h
% Fixed-size arrays and plain loops, so that MATLAB Coder accepts it; opendpd.generateCode copies this text.
n = size(input, 1);
hidden = size(weightHH, 2);
projected = weightIH * input.' + biasIH;          % 3H-by-N, all time steps at once
output = zeros(n, hidden);
for t = 1:n
    gh = weightHH * h + biasHH;
    gi = projected(:, t);
    r = 1 ./ (1 + exp(-(gi(1:hidden) + gh(1:hidden))));
    z = 1 ./ (1 + exp(-(gi(hidden+1:2*hidden) + gh(hidden+1:2*hidden))));
    c = tanh(gi(2*hidden+1:3*hidden) + r .* gh(2*hidden+1:3*hidden));
    h = (1 - z) .* c + z .* h;
    output(t, :) = h.';
end
end
