function config = importOptions(options)
required = ["SampleRate", "Bandwidth", "SegmentSamples"];
missing = required(~ismember(required, string(fieldnames(options))));
if ~isempty(missing)
    error('opendpd:SignalMetadata', ['Supply %s. SampleRate and Bandwidth are in Hz. SegmentSamples is the PSD ' ...
        'segment length in samples: spectral metrics use it as their Welch segment and evaluation restarts a ' ...
        'model''s state at the same interval, so it has no default (Studio''s generated signals use 512-4096).'], ...
        strjoin(missing, ', '));
end
config = struct('sample_rate_hz', options.SampleRate, ...
    'bandwidth_hz', options.Bandwidth, 'nperseg', options.SegmentSamples, ...
    'n_sub_ch', options.Subchannels, 'guard_samples', options.GuardSamples, ...
    'origin', options.Origin, 'amplitude_units', options.AmplitudeUnits);
if strlength(options.Name) > 0
    config.dataset_id = options.Name;
end
if isfield(options, 'Source') && ~isempty(fieldnames(options.Source))
    config.source = options.Source;
end
end
