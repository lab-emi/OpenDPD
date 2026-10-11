function y = hardswish(x)
%HARDSWISH x * min(max(x + 3, 0), 6) / 6, as torch.nn.Hardswish.
y = x .* min(max(x + 3, 0), 6) / 6;
end
