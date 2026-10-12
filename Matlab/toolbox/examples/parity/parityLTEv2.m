function results = parityLTEv2(inputFile, outputFile)
%PARITYLTEV2 MATLAB side of the version 2 LTE cross-validation registered in docs/protocols/waveform-profiles-v2.md.
%   RESULTS = parityLTEv2(INPUTFILE, OUTPUTFILE) reads the same inputs as parityLTE (a MAT file with the signals and a
%   JSON file with their description, written by scripts/matlab_parity.py) and writes a JSON record with the same
%   fields, computed with MathWorks functions only:
%     P1  the LTE waveform from the package symbols (lteOFDMModulate) against the package waveform,
%     P2  adjacent-channel power ratios with comm.ACPR: "default" is the Auto comparator of ofdm-lte20-evm-auto-v2,
%         "aligned" the Hann/50 %/segment-length comparator of ofdm-lte20-evm-v2; object settings as in parityLTE,
%     P3  EVM from an independent implementation of the documented procedure (resample, lteOFDMDemodulate, lteEVM),
%         with the v2 frequency-tolerant acquisition of the registration's "Frequency-tolerant acquisition" section.
%   The only difference from parityLTE is the timing step of P3. parityLTE is kept unchanged as the record of the
%   original (v1) registration. Nothing here calls OpenDPD or Python; budgets are applied by the Python harness.
%
%   Requires Communications Toolbox, LTE Toolbox and Signal Processing Toolbox.
arguments
    inputFile (1,1) string {mustBeFile}
    outputFile (1,1) string
end
description = jsondecode(fileread(replace(inputFile, ".mat", ".json")));
data = load(inputFile);

enb = struct('NDLRB', 100, 'CyclicPrefix', 'Normal', 'Windowing', 0);
info = lteOFDMInfo(enb);
assert(info.SamplingRate == 30.72e6 && info.Nfft == 2048, 'Unexpected LTE numerology');

results = struct();
results.protocol = struct('registration', 'docs/protocols/waveform-profiles-v2.md', ...
    'registration_commit', '8a8c77f594703da8cccf747cda18c905b109a491', 'script', 'parityLTEv2', ...
    'aclr_default_comparator_for', 'ofdm-lte20-evm-auto-v2', 'aclr_aligned_comparator_for', 'ofdm-lte20-evm-v2', ...
    'timing_threshold', 0.3, 'cp_guard_samples', 32, 'max_fine_iterations', 3);
results.environment = environment();
results.p1 = p1Waveform(enb, double(data.symbols), double(data.x_package));
reference = results.p1.reference;
results.p1 = rmfield(results.p1, 'reference');
results.acpr_convention_check = acprConventionCheck();

signals = description.signals;
results.signals = struct();
for k = 1:numel(signals)
    entry = signals(k);
    y = double(data.(entry.id));
    y = y(:);
    fs = entry.sample_rate_hz;
    row = struct('id', entry.id, 'sample_rate_hz', fs);
    row.evm = independentEVM(enb, y, fs, reference, double(data.symbols));
    if isfield(data, entry.id + "_baseband") && fs ~= 30.72e6
        % Diagnostic: the same chain fed with the baseband signal OpenDPD's own resampler produced.
        row.evm_on_opendpd_baseband = independentEVM(enb, double(data.(entry.id + "_baseband")(:)), 30.72e6, ...
            reference, double(data.symbols));
        row.resampler_relative_difference = resamplerDifference(y, fs, double(data.(entry.id + "_baseband")(:)));
    end
    if fs >= 58e6
        row.aclr = aclrAll(y, fs, entry.nperseg);
    end
    if isfield(entry, 'toolbox_frequency_offset') && entry.toolbox_frequency_offset
        row.frequency_offset_toolbox = lteFrequencyOffsetDiagnostic(enb, y, fs);
    end
    results.signals.(entry.id) = row;
end

text = jsonencode(results, PrettyPrint=true);
fid = fopen(outputFile, 'w');
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, text, 'char');
end

% --------------------------------------------------------------------------------------------------------------

function e = environment()
products = ver;
wanted = ["Communications Toolbox", "LTE Toolbox", "Signal Processing Toolbox"];
e = struct('matlab', version, 'release', version('-release'));
for name = wanted
    index = find(strcmp({products.Name}, name), 1);
    key = strrep(char(name), ' ', '_');
    if isempty(index)
        e.(key) = 'not installed';
    else
        e.(key) = products(index).Version;
    end
end
end

function r = p1Waveform(enb, symbols, packageWaveform)
% P1, unchanged from parityLTE. The package symbols are n_symbols-by-1200 in subcarrier order k = -600..-1, +1..+600;
% the LTE resource grid has the same subcarriers as rows (the DC subcarrier is not part of the grid).
grid = symbols.';
x = lteOFDMModulate(enb, grid);
r = struct();
r.reference = x;
r.n_samples_matlab = numel(x);
r.n_samples_package = numel(packageWaveform);
period = numel(packageWaveform);
if numel(x) == period
    correlation = ifft(fft(packageWaveform) .* conj(fft(x)));
    [~, peak] = max(abs(correlation));
    r.peak_lag_samples = peak - 1;
    scale = real(sum(conj(x) .* packageWaveform)) / real(sum(abs(x).^2));
    r.scale = scale;
    r.relative_rms_difference = norm(packageWaveform - scale * x) / norm(packageWaveform);
else
    r.peak_lag_samples = NaN;
    r.scale = NaN;
    r.relative_rms_difference = NaN;
end
end

function r = acprConventionCheck()
% Unchanged from parityLTE: comm.ACPR must report leakage in dB, negative for a weaker adjacent channel.
fs = 122.88e6;
n = 2^18;
rng(11);
f = (-n/2:n/2-1).' * fs / n;
white = complex(randn(n, 1), randn(n, 1));
mainBand = ifft(ifftshift(fftshift(fft(white)) .* (abs(f) <= 9e6)));
adjBand = ifft(ifftshift(fftshift(fft(complex(randn(n, 1), randn(n, 1)))) .* (f >= 11e6 & f <= 29e6)));
mainBand = mainBand / sqrt(mean(abs(mainBand).^2));
adjBand = adjBand / sqrt(mean(abs(adjBand).^2)) * 10^(-40/20);
x = mainBand + adjBand;
acpr = comm.ACPR('SampleRate', fs, 'MainChannelFrequency', 0, 'MainMeasurementBandwidth', 18e6, ...
    'AdjacentChannelOffset', [-20e6 20e6], 'AdjacentMeasurementBandwidth', 18e6, 'PowerUnits', 'dBW');
value = acpr(x);
r = struct('constructed_right_dB', -40, 'measured_left_dB', value(1), 'measured_right_dB', value(2));
r.ok = abs(value(2) - (-40)) < 0.3 && value(1) < -60;
if ~r.ok
    error('parityLTEv2:ACPRConvention', ...
        'comm.ACPR does not report leakage in dB with this sign (right %g dB, left %g dB); stop before comparing.', ...
        value(2), value(1));
end
end

function r = aclrAll(y, fs, nperseg)
% P2, unchanged from parityLTE. Left is the channel at -20 MHz, right at +20 MHz. "default" (comm.ACPR Auto) is the
% comparator of ofdm-lte20-evm-auto-v2; "aligned" (Hann, 50 %, FFT = segment) that of ofdm-lte20-evm-v2.
r = struct();
r.default = acprValues(y, fs, struct());
r.aligned = struct();
r.pwelch = struct();
r.enclosing_rule_replica = struct();
for n = nperseg(:).'
    key = "n" + string(n);
    r.aligned.(key) = acprValues(y, fs, struct('nperseg', n));
    r.pwelch.(key) = pwelchValues(y, fs, n);
    r.enclosing_rule_replica.(key) = pwelchValues(y, fs, n, true);
end
end

function v = acprValues(y, fs, options)
acpr = comm.ACPR('SampleRate', fs, 'MainChannelFrequency', 0, 'MainMeasurementBandwidth', 18e6, ...
    'AdjacentChannelOffset', [-20e6 20e6], 'AdjacentMeasurementBandwidth', 18e6, 'PowerUnits', 'dBW');
if isfield(options, 'nperseg')
    acpr.SpectralEstimation = 'Specify window parameters';
    acpr.SegmentLength = options.nperseg;
    acpr.OverlapPercentage = 50;
    acpr.Window = 'Hann';
    acpr.FFTLength = 'Same as segment length';
end
value = acpr(y);
v = struct('left_dB', value(1), 'right_dB', value(2));
end

function v = pwelchValues(y, fs, nperseg, enclosing)
% Diagnostic, unchanged from parityLTE: Signal Processing Toolbox Welch, Hann, 50 % overlap. By default the band rule
% is the v1 profile's (bins whose centre lies in [lo, hi), times the bin width); with ENCLOSING true it is the
% enclosing rule (from the last bin at or below lo through the first bin at or above hi).
if nargin < 4
    enclosing = false;
end
[p, f] = pwelch(y, hann(nperseg, 'periodic'), nperseg / 2, nperseg, fs, 'centered', 'psd');
width = fs / nperseg;
if enclosing
    band = @(lo, hi) sum(p(find(f <= lo, 1, 'last') : find(f >= hi, 1))) * width;
else
    band = @(lo, hi) sum(p(f >= lo & f < hi)) * width;
end
main = band(-9e6, 9e6);
v = struct('left_dB', 10 * log10(band(-29e6, -11e6) / main), 'right_dB', 10 * log10(band(11e6, 29e6) / main));
end

function d = resamplerDifference(y, fs, opendpdBaseband)
[p, q] = rationalRatio(fs);
if p == q
    converted = y;
else
    converted = resample(y, p, q);
end
d = norm(converted - opendpdBaseband) / norm(opendpdBaseband);
end

function [p, q] = rationalRatio(fs)
g = gcd(30720000, round(fs));
p = 30720000 / g;
q = round(fs) / g;
end

% --- P3 ----------------------------------------------------------------------------------------------------------

function [tau, peak] = circularTiming(w, xref, magnitudes)
% Integer timing from the circular cross-correlation of the window W (one period) with the reference waveform:
% y[n] ~ x[(n + tau) mod period]. With MAGNITUDES true both sides are replaced by their magnitudes (a timing
% candidate only); PEAK is the peak normalised by the two norms.
if magnitudes
    w = abs(w);
    xref = abs(xref);
end
c = ifft(fft(w) .* conj(fft(xref)));
[~, m] = max(abs(c));
tau = mod(-(m - 1), numel(xref));
peak = abs(c(m)) / (norm(w) * norm(xref));
end

function [starts, cpLengths] = completeSymbols(cpPeriod, period, tau, nCapture)
% Zero-based capture positions of the cyclic-prefix start of every complete symbol (prefix and useful part inside
% the capture) for an arbitrary timing TAU and capture length, in capture order. CPPERIOD is the cyclic-prefix
% length of every symbol of one waveform period, which starts at reference sample 0.
nfft = 2048;
cpStart = [0, cumsum(cpPeriod(1:end-1) + nfft)];
starts = [];
cpLengths = [];
for m = floor(tau / period) : ceil((nCapture + tau) / period)
    s = cpStart - tau + m * period;
    keep = s >= 0 & s + cpPeriod + nfft <= nCapture;
    starts = [starts, s(keep)];                                %#ok<AGROW>
    cpLengths = [cpLengths, cpPeriod(keep)];                   %#ok<AGROW>
end
[starts, order] = sort(starts);
cpLengths = cpLengths(order);
end

function [cfo, nUsed] = cyclicPrefixOffset(y30, starts, cpLengths)
% Coarse frequency offset from the cyclic-prefix correlation of the given complete symbols, skipping the first 32
% prefix samples (channel memory of the previous symbol).
nfft = 2048;
accumulator = 0;
for l = 1:numel(starts)
    headSamples = y30(starts(l) + 1 + 32 : starts(l) + cpLengths(l));
    tailSamples = y30(starts(l) + cpLengths(l) + 1 + nfft - cpLengths(l) + 32 : starts(l) + cpLengths(l) + nfft);
    accumulator = accumulator + sum(headSamples .* conj(tailSamples));
end
cfo = -angle(accumulator) / (2 * pi) * 15e3;
nUsed = numel(starts);
end

function r = independentEVM(enb, y, fs, xref, symbolsRowMajor)
% P3, step by step as registered. y is a complex column at rate fs; xref the LTE reference waveform (one period).
fs30 = 30.72e6;
[p, q] = rationalRatio(fs);
if p == q
    y30 = y;
else
    y30 = resample(y, p, q);                                   % step 1
end
period = numel(xref);
subframe = 30720;
nfft = 2048;
cp = double(lteOFDMInfo(enb).CyclicPrefixLengths);
cpPeriod = repmat(cp(:).', 1, period / subframe);

% step 2 (v2): timing. First the v1 complex circular correlation of one period of the capture.
w = zeros(period, 1);
head = min(period, numel(y30));
w(1:head) = y30(1:head);
[tau, peak] = circularTiming(w, xref, false);
acquisition = struct('method', 'complex', 'initial_timing_samples', tau, 'initial_correlation', peak);
if peak < 0.3
    % v2 fallback. Magnitude correlation gives only a candidate timing; the coarse offset from the complete cyclic
    % prefixes at that candidate is removed and the complex correlation repeated. Only its peak (>= 0.3, as v1)
    % establishes the binding and the timing.
    acquisition.method = 'magnitude_candidate_then_complex';
    [candidate, magnitudePeak] = circularTiming(w, xref, true);
    acquisition.candidate_timing_samples = candidate;
    acquisition.magnitude_correlation = magnitudePeak;
    [starts, cpLengths] = completeSymbols(cpPeriod, period, candidate, numel(y30));
    if isempty(starts)
        acquisition.candidate_frequency_offset_hz = NaN;
        acquisition.candidate_symbols = 0;
        r = struct('status', 'missing_reference', 'timing_samples', tau, 'correlation', peak, ...
            'acquisition', acquisition);
        return
    end
    [candidateOffset, nUsed] = cyclicPrefixOffset(y30, starts, cpLengths);
    acquisition.candidate_frequency_offset_hz = candidateOffset;
    acquisition.candidate_symbols = nUsed;
    derotated = w .* exp(-2j * pi * candidateOffset / fs30 * (0:period - 1).');
    [tau, peak] = circularTiming(derotated, xref, false);
    acquisition.final_timing_samples = tau;
    acquisition.final_correlation = peak;
    acquisition.final_timing_equals_candidate = tau == candidate;
end
r = struct('status', 'ok', 'timing_samples', tau, 'correlation', peak, 'acquisition', acquisition);
if r.correlation < 0.3
    % A final normalised complex peak below 0.3 means the signal is not (a distorted copy of) the bound waveform:
    % missing_reference instead of a number.
    r.status = 'missing_reference';
    return
end
if mod(tau, subframe) ~= 0 || mod(numel(y30), subframe) ~= 0
    % lteOFDMDemodulate starts at the first sample: the demodulation below handles captures that start on a
    % subframe boundary and hold whole subframes (as every registered signal does).
    error('parityLTEv2:Alignment', ...
        'This check handles captures that start on a subframe boundary and hold whole subframes (tau = %d).', tau);
end
nSubframes = numel(y30) / subframe;
symbolsPerSubframe = 14;
nReference = size(symbolsRowMajor, 1);
columns = mod((0:symbolsPerSubframe * nSubframes - 1) + symbolsPerSubframe * (tau / subframe), nReference) + 1;
reference = symbolsRowMajor(columns, :).';

% step 3: coarse frequency offset of the original capture from the cyclic prefix of every complete symbol at the
% final timing, skipping the first 32 prefix samples
[starts, cpLengths] = completeSymbols(cpPeriod, period, tau, numel(y30));
cfo = cyclicPrefixOffset(y30, starts, cpLengths);
r.coarse_frequency_offset_hz = cfo;

% step 4: at most three rounds of {correct, demodulate, per-subcarrier least squares, fine offset}
useful = starts + cpLengths;                                   % zero-based start of each useful part
n = (0:numel(y30) - 1).';
times = (useful + nfft / 2) / fs30;
for iteration = 1:3
    corrected = y30 .* exp(-2j * pi * cfo / fs30 * n);
    received = lteOFDMDemodulate(enb, corrected, 1);
    gain = sum(received .* conj(reference), 2) ./ sum(abs(reference).^2, 2);
    equalised = received ./ gain;
    phase = unwrap(angle(sum(equalised .* conj(reference), 1))).';
    fit = polyfit(times(:), phase, 1);
    update = fit(1) / (2 * pi);
    cfo = cfo + update;
    if abs(update) < 1e-3
        break
    end
end
r.frequency_offset_hz = cfo;
r.iterations = iteration;

% step 5
evm = lteEVM(equalised(:), reference(:));
r.evm_rms_percent = 100 * evm.RMS;
r.n_symbols = size(reference, 2);
end

function d = lteFrequencyOffsetDiagnostic(enb, y, fs)
% Diagnostic: the Toolbox's own cyclic-prefix estimate of the capture, which starts on a subframe boundary.
[p, q] = rationalRatio(fs);
d = lteFrequencyOffset(enb, resample(y, p, q));
end
