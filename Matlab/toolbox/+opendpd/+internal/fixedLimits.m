function limits = fixedLimits()
%FIXEDLIMITS What this toolbox accepts in a fixed-point-v1 package, in one place for the reader and the model.
%   Every number is a limit on what an untrusted archive can ask this MATLAB session to hold or compute, set well above what
%   OpenDPD writes (the default package has 6 to 24 hidden units, tables of 4096 and 2048 entries and 72 thousand golden
%   samples) and low enough that a package at the limit still loads in seconds and in a few hundred megabytes.
limits = struct( ...
    'MaxHidden', 512, ...                 % jsondecode holds about 360 bytes per number: the recurrent matrix alone is 0.3 GB here
    'MaxTableEntries', 2^16, ...          % entries of one lookup table
    'MaxGoldenSamples', 2^21, ...         % samples of all golden cases together (with the trace limit: replaying the largest allowed package takes 20-40 s)
    'MaxNesting', 8, ...                  % levels of nesting of any JSON file (OpenDPD writes at most 4)
    'ManifestBytes', 4e6, 'SpecBytes', 1e6, 'WeightsBytes', 32e6, 'SourceBytes', 32e6, 'MetaBytes', 1e6, ...
    'TraceBytes', 160e6, ...              % h_trace.i16: 2 bytes per unit and sample
    'ManifestSeparators', 4000, 'SpecSeparators', 500);
end
