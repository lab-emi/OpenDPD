function results = parityPAmemory(options)
%PARITYPAMEMORY rf.PAmemory against comm.DPD and OpenDPD, as registered in docs/performance/matlab-parity-dpd.md (Amendment 2).
%   RESULTS = parityPAmemory() runs the four registered items and prints them with their budgets:
%     R1  rf.PAmemory on [zeros(Q-1,1); x], outputs Q to the end, against comm.DPD on x (zero initial state);
%     R2  the first Q-1 outputs of rf.PAmemory on x against comm.DPD on x preceded by Q-1 copies of x(1);
%     R3  opendpd.apply of the mp_ls-dpd fixture on one 128-sample segment against rf.PAmemory with
%         model.commCoefficients() on [zeros(Q-1,1); x];
%     R4  a second call of the same rf.PAmemory object on x(129:256) against one call on all of x.
%   Memory polynomial only, Q in {1,2,4,7}, K in {1,3,5}, three signals of 256 samples (seeds 11, 12, 13). The coefficient
%   matrix of a pair (Q,K) is complex randn/(Q*K) from the seed 1000 + 10*Q + K. Nothing is tuned: the budgets are the registered
%   ones and the verdict is computed here from them.
%
%   Requires RF Toolbox and Communications Toolbox, and the OpenDPD toolbox on the path.
arguments
    options.Quiet (1,1) logical = false
end
if ~license('test', 'RF_Toolbox') || ~license('test', 'Communication_Toolbox')
    error('opendpd:parity', 'This comparison needs RF Toolbox and Communications Toolbox.');
end
sizes = [1 1; 1 3; 1 5; 2 1; 2 3; 2 5; 4 1; 4 3; 4 5; 7 1; 7 3; 7 5];       % [Q K]
seeds = [11 12 13];
tolerance = 1e-12;
relative = @(a, b) max(abs(a(:) - b(:))) / max(abs(b(:)));
r1 = []; r2 = []; r4 = []; labels1 = strings(0, 1); labels2 = strings(0, 1); labels4 = strings(0, 1);
for row = 1:size(sizes, 1)
    Q = sizes(row, 1);
    K = sizes(row, 2);
    stream = RandStream('twister', Seed=1000 + 10 * Q + K);
    C = complex(randn(stream, Q, K), randn(stream, Q, K)) / (Q * K);
    for seed = seeds
        s = RandStream('twister', Seed=seed);
        x = 0.3 * complex(randn(s, 256, 1), randn(s, 256, 1));
        label = sprintf('Q=%d K=%d seed=%d', Q, K, seed);

        reference = comm.DPD(PolynomialType='Memory polynomial', Coefficients=C);
        padded = rf.PAmemory(Model='Memory polynomial', CoefficientMatrix=C);
        y = padded([zeros(Q - 1, 1); x]);
        r1(end+1) = relative(y(Q:end), reference(x)); %#ok<AGROW>
        labels1(end+1) = label; %#ok<AGROW>

        if Q >= 2
            bare = rf.PAmemory(Model='Memory polynomial', CoefficientMatrix=C);
            first = bare(x);
            copies = comm.DPD(PolynomialType='Memory polynomial', Coefficients=C);
            z = copies([repmat(x(1), Q - 1, 1); x]);
            r2(end+1) = max(abs(first(1:Q-1) - z(Q:2*Q-2))) / max(abs(z(Q:end))); %#ok<AGROW>
            labels2(end+1) = label; %#ok<AGROW>
        end

        carried = rf.PAmemory(Model='Memory polynomial', CoefficientMatrix=C);
        carried(x(1:128));
        second = carried(x(129:256));
        whole = rf.PAmemory(Model='Memory polynomial', CoefficientMatrix=C);
        w = whole(x);
        r4(end+1) = relative(second, w(129:256)); %#ok<AGROW>
        labels4(end+1) = label; %#ok<AGROW>
    end
end

fixture = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'tests', 'data', 'mp_ls-dpd.opendpd.zip');
model = opendpd.load(fixture);
[C, info] = model.commCoefficients();
Q = info.memory_depth;
r3 = []; labels3 = strings(0, 1);
for seed = seeds
    s = RandStream('twister', Seed=seed);
    x = 0.3 * complex(randn(s, 128, 1), randn(s, 128, 1));
    od = double(opendpd.apply(model, x));
    pa = rf.PAmemory(Model='Memory polynomial', CoefficientMatrix=C);
    y = pa([zeros(Q - 1, 1); double(single(x))]);
    r3(end+1) = relative(y(Q:end), od); %#ok<AGROW>
    labels3(end+1) = sprintf('mp_ls-dpd seed=%d', seed); %#ok<AGROW>
end

items = {'R1', r1, labels1, tolerance; 'R2', r2, labels2, tolerance; 'R3', r3, labels3, 1e-6; 'R4', r4, labels4, tolerance};
results = struct('item', {}, 'cases', {}, 'worst_relative_difference', {}, 'worst_case', {}, 'budget', {}, 'within_budget', {});
for k = 1:size(items, 1)
    [worst, at] = max(items{k, 2});
    results(k) = struct('item', items{k, 1}, 'cases', numel(items{k, 2}), 'worst_relative_difference', worst, ...
        'worst_case', char(items{k, 3}(at)), 'budget', items{k, 4}, 'within_budget', worst <= items{k, 4});
end
v = ver('rf');
if ~options.Quiet
    fprintf('rf.PAmemory against comm.DPD and OpenDPD (MATLAB %s, RF Toolbox %s)\n', version, v(1).Version);
    for k = 1:numel(results)
        r = results(k);
        fprintf('  %s  %2d cases  worst %.3g (%s)  budget %.0e  %s\n', r.item, r.cases, r.worst_relative_difference, ...
            r.worst_case, r.budget, ternary(r.within_budget, 'within', 'OUTSIDE'));
    end
    fprintf('Verdict: %s\n', ternary(all([results.within_budget]), 'every item within its registered budget', 'NOT passed as registered'));
end
end

function value = ternary(condition, a, b)
if condition
    value = a;
else
    value = b;
end
end
