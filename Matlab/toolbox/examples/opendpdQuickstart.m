function summary = opendpdQuickstart()
%OPENDPDQUICKSTART Train and apply a small GRU on clearly synthetic PA data.
% First select Python with opendpd.setup(PythonExecutable="...").
% Requires base MATLAB and the Python OpenDPD checkout; no RF hardware.
workspace = string(tempname) + "-opendpd";
p = opendpd.openProject(workspace);
cleanup = onCleanup(@() opendpd.closeProject(p, StopService=true)); %#ok<NASGU>

fs = 80e6;
bw = 20e6;
n = 4096;
rng(42);
f = (-n/2:n/2-1).' * fs/n;
spectrum = complex(randn(n,1), randn(n,1));
spectrum(abs(f) > bw/2) = 0;
x = ifft(ifftshift(spectrum));
x = single(0.2 * x / sqrt(mean(abs(x).^2)));
delayed = [complex(single(0)); x(1:end-1)];
y = x - 0.25 * x .* abs(x).^2 + 0.03 * delayed;

ds = opendpd.importIQ(p, x, y, SampleRate=fs, Bandwidth=bw, ...
    Name="synthetic-demo", Origin="synthetic", AmplitudeUnits="normalized", SegmentSamples=128);
training = struct('epochs', 2, 'frame_length', 32, 'frame_stride', 32, ...
    'batch_size', 16, 'batch_size_eval', 16);
parameters = struct('hidden_size', 8);
pa = opendpd.wait(opendpd.trainPA(p, ds, ModelParameters=parameters, Training=training));
dpd = opendpd.wait(opendpd.trainDPD(p, ds, PA=pa, ModelParameters=parameters, Training=training));

% Manifest boundaries are zero-based, half-open; MATLAB indices are one-based.
range = ds.split.boundaries.test;
xTest = x(range(1)+1:range(2));
[u, info] = opendpd.apply(dpd, xTest);
exported = opendpd.wait(opendpd.runDPD(dpd));
report = opendpd.result(dpd); % segmented test evaluation matching apply above
outputFile = fullfile(p.Workspace, 'matlab-dpd-output.mat');
save(outputFile, 'xTest', 'u', 'info', 'fs', 'report', '-v7');

summary = struct('workspace', p.Workspace, 'pa_run', pa.ID, 'dpd_run', dpd.ID, ...
    'export_run', exported.ID, 'waveform_file', outputFile);
disp(summary);
% Reopen with p = opendpd.openProject(summary.workspace); opendpd.openStudio(p).
% This small run checks the workflow; it is not a performance benchmark.
end
