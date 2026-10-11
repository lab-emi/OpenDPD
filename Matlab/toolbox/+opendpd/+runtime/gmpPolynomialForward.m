function y = gmpPolynomialForward(x, coefficients, p)
%GMPPOLYNOMIALFORWARD Generalised memory polynomial (gmp_ls) with zero history and zero future outside the segment.
% p has Ka La Kb Lb Mb Kc Lc Mc. The coefficient vector lists the aligned terms x(n-l)|x(n-l)|^k (k, then l),
% the lagging terms x(n-l)|x(n-l-m)|^k (k = 1..Kb, l, m = 1..Mb) and the leading terms x(n-l)|x(n-l+m)|^k
% (k = 1..Kc, l, m = 1..Mc), in that order. Mirrors opendpd.core.polynomial.gmp_basis.
x = x(:);
y = complex(zeros(numel(x), 1));
index = 0;
for k = 0:p.Ka-1
    for l = 0:p.La-1
        index = index + 1;
        d = opendpd.runtime.delaySignal(x, l);
        y = y + coefficients(index) * (d .* abs(d) .^ k);
    end
end
for k = 1:p.Kb
    for l = 0:p.Lb-1
        for m = 1:p.Mb
            index = index + 1;
            y = y + coefficients(index) * (opendpd.runtime.delaySignal(x, l) .* abs(opendpd.runtime.delaySignal(x, l + m)) .^ k);
        end
    end
end
for k = 1:p.Kc
    for l = 0:p.Lc-1
        for m = 1:p.Mc
            index = index + 1;
            y = y + coefficients(index) * (opendpd.runtime.delaySignal(x, l) .* abs(opendpd.runtime.delaySignal(x, l - m)) .^ k);
        end
    end
end
if index ~= numel(coefficients)
    error('opendpd:Package', 'gmp_ls expects %d coefficients, the package has %d.', index, numel(coefficients));
end
end
