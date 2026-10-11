% OpenDPD Toolbox for MATLAB - for the unreleased OpenDPD 2.4.0.
% Version 2.4.0
%
% GUI:         studio, disconnect, help, openStudio (also Apps > OpenDPDStudio).
% Environment: setup, doctor, openProject, closeProject.
% Data:        importIQ, importMAT.
% Training:    trainPA, trainDPD, submit, getRun, status, wait, cancel.
% One call:   fit (capture -> PA and DPD models; Python runs as a separate process, no pyenv).
% Results:     result, apply, runDPD.
% Metrics:     waveform, metrics.evm, metrics.aclr, metrics.evaluate (no project or server needed).
% Models:      export, load, verify, Model (a loaded package); apply(model, x) and model(chunk) run it in plain MATLAB.
% Fixed point: FixedModel (load a fixed-point-v1 deployment package; verify replays its golden vectors bit for bit).
% Code:        generateCode (a model as a standalone class for Simulink MATLAB System blocks and MATLAB Coder).
% Lab:         lab.Session (supervised measurement with RF-off interlock), lab.MockInstrument (dry run, emits nothing).
%
% All functions use the opendpd namespace, for example opendpd.doctor().
% Start with README.md and examples/opendpdQuickstart.m.
