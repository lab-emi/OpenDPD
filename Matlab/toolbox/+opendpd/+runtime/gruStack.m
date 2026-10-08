function [h, state] = gruStack(features, weightIH, weightHH, biasIH, biasHH, state)
%GRUSTACK A torch.nn.GRU stack (batch of one, gate order r, z, n) over FEATURES (N-by-F).
% weightIH{l} is 3H-by-in, weightHH{l} is 3H-by-H, biasIH{l} and biasHH{l} are 3H-by-1 (zeros when the model has no bias).
% STATE is H-by-L (one column per layer). Returns the last layer's outputs (N-by-H) and the final STATE.
%   r = sigmoid(Wir x + bir + Whr h + bhr)   z = sigmoid(Wiz x + biz + Whz h + bhz)
%   n = tanh(Win x + bin + r .* (Whn h + bhn))   h' = (1 - z) .* n + z .* h
layers = numel(weightIH);
input = features.';                       % F-by-N
n = size(input, 2);
hidden = size(weightHH{1}, 2);
for l = 1:layers
    projected = weightIH{l} * input + biasIH{l};          % 3H-by-N, all time steps at once
    output = zeros(hidden, n);
    h = state(:, l);
    whh = weightHH{l};
    bhh = biasHH{l};
    for t = 1:n
        gh = whh * h + bhh;
        gi = projected(:, t);
        r = 1 ./ (1 + exp(-(gi(1:hidden) + gh(1:hidden))));
        z = 1 ./ (1 + exp(-(gi(hidden+1:2*hidden) + gh(hidden+1:2*hidden))));
        c = tanh(gi(2*hidden+1:3*hidden) + r .* gh(2*hidden+1:3*hidden));
        h = (1 - z) .* c + z .* h;
        output(:, t) = h;
    end
    state(:, l) = h;
    input = output;
end
h = input.';
end
