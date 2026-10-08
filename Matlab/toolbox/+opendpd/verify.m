function report = verify(model)
%VERIFY Run a loaded model package on its golden test vector and report the error against OpenDPD's outputs.
% report = opendpd.verify(model) returns passed, tolerance_abs, offline_max_abs_error and, for models with a streaming
% variant, streaming_max_abs_error (NaN otherwise). The golden outputs are what opendpd.apply produced in Python
% for the same input, so a pass shows that this MATLAB release computes the model the way OpenDPD does.
arguments
    model (1,1) opendpd.Model
end
report = model.verify();
end
