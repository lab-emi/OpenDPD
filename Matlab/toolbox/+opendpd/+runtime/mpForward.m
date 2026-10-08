function y = mpForward(x, coefficients, K, Q)
%MPFORWARD Memory polynomial with zero history: y(n) = sum_k sum_q w(k*Q+q+1) x(n-q) |x(n-q)|^k.
% K envelope orders k = 0..K-1, Q lags q = 0..Q-1; the coefficient vector lists the lag fastest, so
% reshape(w, Q, K) is the Coefficients matrix of comm.DPD. Mirrors opendpd.core.polynomial.mp_basis.
x = x(:);
y = complex(zeros(numel(x), 1));
for q = 0:Q-1
    d = opendpd.runtime.delaySignal(x, q);
    a = abs(d);
    for k = 0:K-1
        y = y + coefficients(k * Q + q + 1) * (d .* a .^ k);
    end
end
end
