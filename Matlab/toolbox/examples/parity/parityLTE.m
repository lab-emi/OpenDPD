function results = parityLTE(inputFile, outputFile)
%PARITYLTE MATLAB side of the LTE cross-validation registered in docs/performance/matlab-parity.md.
%   RESULTS = parityLTE(INPUTFILE, OUTPUTFILE) reads the signals written by scripts/matlab_parity.py
%   (a MAT file with the signals and a JSON file with their description) and computes, with MathWorks
%   functions only,
%     P1  the LTE waveform from the package symbols (lteOFDMModulate) against the package waveform,
%     P2  adjacent-channel power ratios with comm.ACPR (default procedure, and estimator-aligned),
%     P3  EVM from an independent implementation of the documented procedure (resample, lteOFDMDemodulate, lteEVM).
%   RESULTS is also written as JSON to OUTPUTFILE. Nothing here calls OpenDPD; the comparison with OpenDPD's
%   numbers, and the budgets, are applied by scripts/matlab_parity.py.
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
% P1. The package symbols are n_symbols-by-1200 in subcarrier order k = -600..-1, +1..+600; the LTE resource
% grid has the same subcarriers as rows (the DC subcarrier is not part of the grid).
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
% comm.ACPR must report leakage in dB, negative for a weaker adjacent channel, before it is used for anything.
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
    error('parityLTE:ACPRConvention', ...
        'comm.ACPR does not report leakage in dB with this sign (right %g dB, left %g dB); stop before comparing.', ...
        value(2), value(1));
end
end

function r = aclrAll(y, fs, nperseg)
% P2. Left is the channel at -20 MHz, right at +20 MHz.
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
% Diagnostic: Signal Processing Toolbox Welch with the settings of the OpenDPD profile. By default the band rule is
% the profile's (bins whose centre frequency lies in [lo, hi), times the bin width). With ENCLOSING true the rule is
% the one comm.ACPR's code applies: from the last bin at or below lo through the first bin at or above hi.
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

% step 2: integer timing from the circular cross-correlation with the reference waveform
w = zeros(period, 1);
head = min(period, numel(y30));
w(1:head) = y30(1:head);
c = ifft(fft(w) .* conj(fft(xref)));
[~, m] = max(abs(c));
tau = mod(-(m - 1), period);                                   % y[n] ~ x[(n + tau) mod period]
r = struct('status', 'ok', 'timing_samples', tau, 'correlation', abs(c(m)) / (norm(w) * norm(xref)));
if r.correlation < 0.3
    % Protocol section 2, step 2: a normalised peak below 0.3 means the signal is not (a distorted copy of) the
    % bound waveform, and the profile reports missing_reference instead of a number.
    r.status = 'missing_reference';
    return
end
if mod(tau, subframe) ~= 0 || mod(numel(y30), subframe) ~= 0
    error('parityLTE:Alignment', ...
        'This check handles captures that start on a subframe boundary and hold whole subframes (tau = %d).', tau);
end
nSubframes = numel(y30) / subframe;
symbolsPerSubframe = 14;
reference = circshift(symbolsRowMajor.', -symbolsPerSubframe * (tau / subframe), 2);
reference = reference(:, 1:symbolsPerSubframe * nSubframes);

% symbol layout of the capture, from the LTE numerology
cp = double(lteOFDMInfo(enb).CyclicPrefixLengths);
cpAll = repmat(cp(:).', 1, nSubframes);
nfft = 2048;
cpStart = [0, cumsum(cpAll(1:end-1) + nfft)];
useful = cpStart + cpAll;                                      % zero-based start of each useful part

% step 3: coarse frequency offset from the cyclic prefix, skipping its first 32 samples
accumulator = 0;
for l = 1:numel(cpAll)
    headSamples = y30(cpStart(l) + 1 + 32 : cpStart(l) + cpAll(l));
    tailSamples = y30(useful(l) + 1 + nfft - cpAll(l) + 32 : useful(l) + nfft);
    accumulator = accumulator + sum(headSamples .* conj(tailSamples));
end
cfo = -angle(accumulator) / (2 * pi) * 15e3;
r.coarse_frequency_offset_hz = cfo;

% step 4: at most three rounds of {correct, demodulate, per-subcarrier least squares, fine offset}
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
