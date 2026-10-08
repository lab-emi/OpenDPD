function w = waveform(options)
%WAVEFORM The OpenDPD LTE-20 test waveform with known symbols (ofdm-lte20-v1), regenerated from its seed.
% w = opendpd.waveform(Seed=1, Subframes=10) returns a struct with
%   x          complex column vector, 30.72 MS/s, unit average power: the waveform to play (loop it)
%   Symbols    complex matrix (symbols x 1200): the reference symbol on every occupied subcarrier
%              k = -600..-1, +1..+600 of each OFDM symbol, in the order lteOFDMModulate expects after mapping
%   SampleRate 30.72e6
%   Seed, Subframes, SHA256   identify the waveform; OpenDPD datasets bound to it carry the same hash
% EVM of a capture needs the same Seed and Subframes: pass w to opendpd.metrics.evm. The waveform is a test
% signal with known symbols, not a conformance signal. No server or project is needed.
arguments
    options.Seed (1,1) double {mustBeInteger, mustBeNonnegative, mustBeLessThanOrEqual(options.Seed, 4294967295)} = 1
    options.Subframes (1,1) double {mustBeInteger, mustBePositive, mustBeLessThanOrEqual(options.Subframes, 1000)} = 10
end
module = bridge();
output = module.lte_waveform(options.Seed, options.Subframes);
iq = double(output{1});
symbols = double(output{2});
metadata = jsondecode(char(output{3}));
w = struct('x', complex(iq(:,1), iq(:,2)), ...
    'Symbols', reshape(complex(symbols(:,1), symbols(:,2)), metadata.occupied_subcarriers, metadata.n_symbols).', ...
    'SampleRate', metadata.sample_rate_hz, 'Seed', metadata.seed, 'Subframes', metadata.n_subframes, ...
    'SHA256', metadata.sha256);
end
