function exported = runDPD(dpd)
%RUNDPD Queue the standard test-split DPD export and surrogate evaluation.
% After wait, the waveform and metadata are registered artifacts in Studio.
arguments
    dpd (1,1) opendpd.Job
end
exported = opendpd.Job(dpd.Project, dpd.Project.Backend.run_dpd(dpd.Backend));
end
