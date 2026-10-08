function config = importOptions(options)
if ~isfield(options, 'SampleRate') || ~isfield(options, 'Bandwidth')
    error('opendpd:SignalMetadata', 'Supply SampleRate and Bandwidth in Hz.');
end
config = struct('sample_rate_hz', options.SampleRate, ...
    'bandwidth_hz', options.Bandwidth, 'nperseg', options.SegmentSamples, ...
    'n_sub_ch', options.Subchannels, 'guard_samples', options.GuardSamples, ...
    'origin', options.Origin, 'amplitude_units', options.AmplitudeUnits);
if strlength(options.Name) > 0
    config.dataset_id = options.Name;
end
end
