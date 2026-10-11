function report = verify(model)
%VERIFY Run a loaded model package on its golden test vector and report the error against OpenDPD's outputs.
% For an opendpd.Model (opendpd-model-v1): report = opendpd.verify(model) returns passed, tolerance_abs,
% offline_max_abs_error and, for models with a streaming variant, streaming_max_abs_error (NaN otherwise). The golden
% outputs are what opendpd.apply produced in Python for the same input, so a pass shows that this MATLAB release computes
% the model the way OpenDPD does.
% For an opendpd.FixedModel (fixed-point-v1): the six golden vectors are replayed and every output sample and every state
% step must be equal, bit for bit. The report has status ("bit_exact" or "mismatch"), passed, cases_checked, samples_checked
% and, for a mismatch, mismatch_case, mismatch_sample (1-based) and mismatch_signal ("h" before "y" for the same step).
arguments
    model (1,1) {mustBeA(model, ["opendpd.Model", "opendpd.FixedModel"])}
end
report = model.verify();
end
