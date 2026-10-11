function features = tresFeatures(iq)
%TRESFEATURES The six tres_gru input features of IQ (N-by-2): I, Q, |x|, |x|^3, I(n+1), Q(n+1).
% The "next" sample of the last sample is the first sample of the segment (torch.roll over the segment).
amplitude = sqrt(iq(:, 1) .^ 2 + iq(:, 2) .^ 2);
next = circshift(iq, -1, 1);
features = [iq, amplitude, amplitude .^ 3, next];
end
