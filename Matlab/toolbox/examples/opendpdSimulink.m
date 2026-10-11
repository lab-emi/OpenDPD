%% OpenDPD in Simulink: a DPD and a PA as MATLAB System blocks
% Takes two opendpd-model-v1 packages (a DPD and the PA model it was trained against), writes each as a standalone MATLAB
% System object with opendpd.generateCode, builds the Simulink chain  samples -> DPD -> PA -> output,  runs it one sample
% per time step and compares the result with opendpd.apply in plain MATLAB.
%
% With no variables set it uses the two small packages that ship with the toolbox tests. Their weights are random, so the
% chain demonstrates the plumbing and linearises nothing. To use your own pair, set the variables first:
%   dpdPackage = "apa-dpd.opendpd.zip"; paPackage = "apa-pa.opendpd.zip"; opendpdSimulink
% (both written by opendpd.export). Needs Simulink. Python is not used.
if ~exist('dpdPackage', 'var') || ~exist('paPackage', 'var')
    testData = fullfile(fileparts(fileparts(which('opendpd.load'))), 'tests', 'data');
    dpdPackage = fullfile(testData, 'gru-dpd.opendpd.zip');
    paPackage = fullfile(testData, 'gru-pa.opendpd.zip');
end
assert(license('test', 'Simulink') && ~isempty(which('new_system')), 'This example needs Simulink.');

%% Load the packages and check them on this MATLAB release
dpd = opendpd.load(dpdPackage);
pa = opendpd.load(paPackage);
assert(opendpd.verify(dpd).passed && opendpd.verify(pa).passed, 'a package failed its golden test');
assert(dpd.Manifest.execution.streaming_stateful.available && pa.Manifest.execution.streaming_stateful.available, ...
    'one sample per time step needs a model with a streaming variant (gru or gmp)');

%% Write the System objects
% "streaming" carries the recurrent state from one time step to the next. The default, "offline_segmented", restarts the
% state every nperseg samples, which is how OpenDPD scored the run: use it for whole frames, not for single samples.
folder = fullfile(tempname, 'opendpd-simulink');
dpdFiles = opendpd.generateCode(dpd, folder, Name="ChainDpd", Execution="streaming");
paFiles = opendpd.generateCode(pa, folder, Name="ChainPa", Execution="streaming");
addpath(folder);
fprintf('Wrote %s\n', folder);
assert(ChainDpdCheck().passed && ChainPaCheck().passed);      % the generated classes pass the golden test too

%% A test waveform in the units the DPD was trained with
rng(1);
count = 400;
level = dpd.Manifest.scaling.train_input.rms;
x = single(level * complex(randn(count, 1), randn(count, 1)) / sqrt(2));

%% The Simulink chain, one sample per time step
mdl = 'opendpdChain';
if bdIsLoaded(mdl)
    close_system(mdl, 0);
end
new_system(mdl);
assignin(get_param(mdl, 'ModelWorkspace'), 'samples', timeseries(x, (0:count - 1).'));
add_block('simulink/Sources/From Workspace', [mdl '/Samples'], 'VariableName', 'samples', ...
    'OutputAfterFinalValue', 'Holding final value', 'SampleTime', '1');
add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/DPD'], 'System', dpdFiles.Name);
add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/PA'], 'System', paFiles.Name);
add_block('simulink/Sinks/To Workspace', [mdl '/Output'], 'VariableName', 'output', 'SaveFormat', 'Array');
add_line(mdl, 'Samples/1', 'DPD/1');
add_line(mdl, 'DPD/1', 'PA/1');
add_line(mdl, 'PA/1', 'Output/1');
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', 'FixedStep', '1', 'StopTime', num2str(count - 1));
save_system(mdl, fullfile(folder, [mdl '.slx']));
fprintf('Saved the model as %s\n', fullfile(folder, [mdl '.slx']));

for simulateUsing = {'Interpreted execution', 'Code generation'}
    set_param([mdl '/DPD'], 'SimulateUsing', simulateUsing{1});
    set_param([mdl '/PA'], 'SimulateUsing', simulateUsing{1});
    simulated = sim(mdl);
    y = double(simulated.output(:));
    % the same chain in plain MATLAB
    reference = double(opendpd.apply(pa, opendpd.apply(dpd, x, Execution="streaming"), Execution="streaming"));
    fprintf('%-22s largest difference from opendpd.apply: %.2g (the signal is %.2g rms)\n', simulateUsing{1}, ...
        max(abs(y - reference)), rms(abs(reference)));
    assert(max(abs(y - reference)) < 1e-6, 'the Simulink chain differs from opendpd.apply');
end
close_system(mdl, 0);                 % the saved copy in the folder above is the model to open
