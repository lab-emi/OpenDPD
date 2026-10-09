classdef TestFixedPoint < matlab.unittest.TestCase
    % A fixed-point-v1 deployment package run by plain MATLAB, bit for bit.
    %   * the six golden vectors of two real packages (the default specification and one that differs in every format) are
    %     reproduced exactly, in outputs and in the state after every sample;
    %   * the kernel equals an independent integer implementation of the specification on random formats and weights;
    %   * a package is data: hostile or damaged ones are refused with a named error, and a golden vector that was altered is
    %     reported at the sample and the signal where it differs;
    %   * the kernel builds with MATLAB Coder and the MEX function reproduces the golden vector.
    % Pure MATLAB except the Coder test; no Python.
    properties (TestParameter)
        package = struct('default', 'gru-pa', 'custom', 'gru-pa-custom')
        seed = num2cell(1:24)
        badPath = struct('parent', "../opendpd_escape.txt", 'absolute', "/tmp/opendpd_escape.txt", ...
            'inner_dotdot', "golden/../weights.json", 'directory', "weights.json/", 'windows', "C:\opendpd_escape.txt", ...
            'trailing_newline', "golden/normal/x.i16" + newline, 'extra_c_file', "c/evil.c", 'extra_golden_file', "golden/normal/extra.bin", 'bad_case_name', "golden/Bad Case/x.i16", ...
            'long_case_name', "golden/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/x.i16", 'newline', "weights.json" + newline)
        shortGolden = struct('inputs', "x.i16", 'outputs', "y.i16", 'trace', "h_trace.i16", 'final', "h_final.i16")
        modified = struct('weights', "weights.json", 'spec', "spec.json", 'readme', "README.md", 'c_source', "c/gru_fixed.c", ...
            'c_header', "c/gru_fixed.h", 'harness', "c/harness.c", 'golden_input', "golden/normal/x.i16", ...
            'golden_output', "golden/extreme/y.i16", 'golden_trace', "golden/saturation/h_trace.i16", ...
            'golden_meta', "golden/all_zero/meta.json")
        badSpec = struct( ...
            'other_specification', {{'"spec_id": "fixed-point-v1"', '"spec_id": "fixed-point-v9"'}}, ...
            'other_execution', {{'"model_key": "gru_stream"', '"model_key": "gru_offline"'}}, ...
            'other_rounding_rule', {{'round half up', 'round half down'}}, ...
            'other_saturation_rule', {{'every stored quantity saturates', 'every stored quantity wraps'}}, ...
            'input_word_too_wide', {{'("x": \{\s*"bits": )16', '$1 17'}}, ...
            'pre_activation_too_wide', {{'("pre": \{\s*"bits": )32', '$1 33'}}, ...
            'accumulator_beyond_double', {{'"accumulator_bits": 48', '"accumulator_bits": 60'}}, ...
            'weights_too_wide', {{'"weight_bits": 16', '"weight_bits": 17'}}, ...
            'table_index_fraction', {{'("sigmoid": \{\s*"function": "sigmoid",\s*"range": 8.0,\s*"index_frac": )8', '$1 17'}}, ...
            'table_range_not_whole_entries', {{'"range": 8.0', '"range": 7.3'}}, ...
            'sigmoid_table_declared_as_tanh', {{'"function": "sigmoid"', '"function": "tanh"'}})
        % values of the wrong type that MATLAB would happily compute with: a logical, a one-character string that equals a
        % number, an array of character codes that equals a string, null and {} where text belongs
        typeConfusion = struct( ...
            'range_is_true', {{'spec', '"range": 8.0', '"range": true'}}, ...
            'rounding_rule_is_character_codes', {{'spec', '"rounding": "[^"]*"', '"rounding": [114,111,117,110,100]'}}, ...
            'specification_id_is_character_codes', {{'spec', '"spec_id": "fixed-point-v1"', ...
                '"spec_id": [102,105,120,101,100,45,112,111,105,110,116,45,118,49]'}}, ...
            'specification_id_is_null', {{'spec', '"spec_id": "fixed-point-v1"', '"spec_id": null'}}, ...
            'specification_id_is_an_object', {{'spec', '"spec_id": "fixed-point-v1"', '"spec_id": {}'}}, ...
            'hidden_is_a_character', {{'weights', '"hidden": 6', '"hidden": "\\u0006"'}}, ...
            'inputs_is_a_character', {{'weights', '"inputs": 2', '"inputs": "\\u0002"'}}, ...
            'bias_fraction_is_a_character', {{'weights', '"bias": 20', '"bias": "\\u0014"'}}, ...
            'weight_fraction_is_a_character', {{'weights', '("fractions": \{\s*"w_ih": )15', '$1 "\\u000f"'}}, ...
            'gate_order_is_character_codes', {{'weights', '"gate_order": \[[^\]]*\]', '"gate_order": [114,122,110]'}}, ...
            'hidden_size_is_a_character', {{'manifest', '"hidden_size": 6', '"hidden_size": "\\u0006"'}}, ...
            'tensor_fraction_is_a_character', {{'manifest', ...
                '("name": "w_ih",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )15', '$1 "\\u000f"'}}, ...
            'resets_are_a_character', {{'manifest', '"resets_at": \[\s*0,\s*256,\s*512,\s*768\s*\]', '"resets_at": "\\u0001"'}})
        % the manifest holds the members opendpd/schemas/fixed_point.py defines and no others
        undefinedMember = struct( ...
            'top_level', {{'("schema_version": 1,)', '$1 "extra": 1,'}}, ...
            'in_every_golden_case', {{'("case_id": "[a-z_]+",)', '$1 "extra": 1,'}}, ...
            'in_every_tensor', {{'("name": "[a-z_]+",)', '$1 "extra": 1,'}}, ...
            'in_the_verification', {{'("backend": "c99",)', '$1 "extra": 1,'}}, ...
            'a_renamed_member', {{'"created_at":', '"created_on":'}})
        % manifest text that is not what the format allows, and what the refusal must say
        badManifestValue = struct( ...
            'other_schema_version', {{'"schema_version": 1', '"schema_version": 2', 'schema_version must be a whole number from 1 to 1'}}, ...
            'creation_time_is_not_a_time', {{'"created_at": "[^"]*"', '"created_at": "yesterday"', 'created_at is missing or has an unexpected form'}}, ...
            'model_key_with_a_space', {{'("run_id": "[^"]*",\s*"model_key": )"gru"', '$1 "has space"', 'model_key is missing or has an unexpected form'}}, ...
            'run_id_that_is_a_path', {{'"run_id": "[^"]*"', '"run_id": "../../x"', 'run_id is missing or has an unexpected form'}}, ...
            'weights_hash_is_not_a_hash', {{'"weights_sha256": "[0-9a-f]{64}"', '"weights_sha256": "abc"', 'weights_sha256 is missing or has an unexpected form'}}, ...
            'hidden_size_beyond_the_limit', {{'"hidden_size": 6', '"hidden_size": 513', 'hidden_size must be a whole number from 1 to 512'}}, ...
            'files_is_a_number', {{'("files": )\{[^}]*\}', '$1 5', 'files must be an object that maps every file'}}, ...
            'report_is_a_number', {{'("report": )\{.*?\}(,\s*"files")', '$1 5$2', 'report must be an object'}}, ...
            'software_is_a_number', {{'("software": )\{[^}]*\}', '$1 5', 'software must be an object'}}, ...
            'software_is_a_list', {{'("software": )\{[^}]*\}', '$1[1,2]', 'software must be an object'}}, ...
            'no_tensors', {{'("tensors": )\[.*?\](,\s*"golden")', '$1[]$2', 'must describe the six quantised tensors'}}, ...
            'a_tensor_the_format_does_not_know', {{'"name": "w_ih"', '"name": "w_xx"', 'unknown or listed twice'}}, ...
            'a_tensor_listed_twice', {{'"name": "w_hh"', '"name": "w_ih"', 'unknown or listed twice'}}, ...
            'a_tensor_shape_the_hidden_size_does_not_imply', {{'("name": "w_ih",\s*"shape": \[\s*18,\s*)2', '$1 3', 'shape other than the one the hidden size implies'}}, ...
            'a_tensor_shape_that_is_text', {{'("name": "w_ih",\s*"shape": )\[[^\]]*\]', '$1"x"', 'shape other than the one the hidden size implies'}}, ...
            'tensor_bits_are_text', {{'("name": "w_ih",\s*"shape": \[[^\]]*\],\s*"bits": )\d+', '$1 "x"', 'tensors[].bits must be a whole number'}}, ...
            'tensor_fraction_is_63', {{'("name": "w_ih",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )\d+', '$1 63', 'tensors[].frac must be a whole number from 0 to 62'}}, ...
            'tensor_peak_is_text', {{'("name": "w_ih",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": \d+,\s*"max_abs_float": )[^,]*', '$1 "x"', 'tensors[].max_abs_float must be a finite number'}}, ...
            'tensor_saturation_count_is_negative', {{'("name": "w_ih",.*?"saturated": )\d+', '$1 -1', 'tensors[].saturated must be a whole number'}}, ...
            'no_golden_cases', {{'("golden": )\[.*?\](,\s*"verification")', '$1[]$2', 'must index exactly the six golden vectors'}}, ...
            'a_case_the_specification_does_not_name', {{'"case_id": "extreme"', '"case_id": "extremes"', 'not one of'}}, ...
            'a_case_listed_twice', {{'"case_id": "extreme"', '"case_id": "normal"', 'each of the six golden vectors once'}}, ...
            'case_description_is_a_number', {{'("case_id": "normal",\s*"description": )"[^"]*"', '$1 5', 'golden[].description is missing or has an unexpected form'}}, ...
            'case_description_is_too_long', {{'("case_id": "normal",\s*"description": ")[^"]*', ['$1' repmat('x', 1, 401)], 'golden[].description is missing or has an unexpected form'}}, ...
            'case_has_no_samples', {{'("case_id": "normal",\s*"description": "[^"]*",\s*"n_samples": )\d+', '$1 0', 'golden[].n_samples must be a whole number from 1 to'}}, ...
            'case_has_half_a_sample', {{'("case_id": "normal",\s*"description": "[^"]*",\s*"n_samples": )\d+', '$1 1.5', 'golden[].n_samples must be a whole number from 1 to'}}, ...
            'case_has_too_many_samples', {{'("case_id": "normal",\s*"description": "[^"]*",\s*"n_samples": )\d+', '$1 3000000', 'golden[].n_samples must be a whole number from 1 to'}}, ...
            'case_resets_are_text', {{'("case_id": "normal",.*?"resets_at": )\[\]', '$1"x"', 'resets_at must be a list of numbers'}}, ...
            'case_input_hash_is_short', {{'("case_id": "normal",.*?"input_sha256": )"[0-9a-f]{64}"', '$1"abc"', 'golden[].input_sha256 is missing or has an unexpected form'}}, ...
            'case_output_hash_is_short', {{'("case_id": "normal",.*?"output_sha256": )"[0-9a-f]{64}"', '$1"abc"', 'golden[].output_sha256 is missing or has an unexpected form'}}, ...
            'case_state_hash_is_short', {{'("case_id": "normal",.*?"state_sha256": )"[0-9a-f]{64}"', '$1"abc"', 'golden[].state_sha256 is missing or has an unexpected form'}}, ...
            'case_trace_hash_is_short', {{'("case_id": "normal",.*?"trace_sha256": )"[0-9a-f]{64}"', '$1"abc"', 'golden[].trace_sha256 is missing or has an unexpected form'}}, ...
            'verification_backend_has_a_bad_character', {{'"backend": "c99"', '"backend": "c99!"', 'verification.backend is missing or has an unexpected form'}}, ...
            'verification_backend_is_too_long', {{'"backend": "c99"', ['"backend": "' repmat('c', 1, 41) '"'], 'verification.backend is missing or has an unexpected form'}}, ...
            'verification_status_is_unknown', {{'"status": "bit_exact"', '"status": "perfect"', 'verification status is not one of'}}, ...
            'verification_checked_a_negative_number_of_cases', {{'"cases_checked": 6', '"cases_checked": -1', 'verification.cases_checked must be a whole number from 0 to 6'}}, ...
            'more_cases_checked_than_exist', {{'"cases_checked": 6', '"cases_checked": 7', 'verification.cases_checked must be a whole number from 0 to 6'}}, ...
            'verification_compiler_is_a_number', {{'"compiler": "[^"]*"', '"compiler": 5', 'verification.compiler must be text or null'}}, ...
            'verification_detail_is_a_number', {{'("verification": \{.*?"detail": )"[^"]*"', '$1 5', 'verification.detail must be text or null'}}, ...
            'verification_detail_is_too_long', {{'("verification": \{.*?"detail": ")[^"]*', ['$1' repmat('d', 1, 2001)], 'verification.detail must be text or null'}}, ...
            'verification_mismatch_case_is_a_list', {{'"mismatch_case": null', '"mismatch_case": ["a","b"]', 'verification.mismatch_case must be text or null'}}, ...
            'verification_mismatch_signal_is_a_number', {{'"mismatch_signal": null', '"mismatch_signal": 5', 'verification.mismatch_signal must be text or null'}}, ...
            'verification_mismatch_step_is_text', {{'"mismatch_step": null', '"mismatch_step": "x"', 'verification.mismatch_step must be a number or null'}}, ...
            'verification_mismatch_step_is_a_list', {{'"mismatch_step": null', '"mismatch_step": [1,2]', 'verification.mismatch_step must be a number or null'}})
        % numbers outside the range the format allows; the last element is what the refusal must say
        outOfRange = struct( ...
            'weight_fraction_63', {{'weights', '("fractions": \{\s*"w_ih": )15', '$1 63', 'fractions.w_ih must be a whole number from 0 to 62'}}, ...
            'hh_fraction_negative', {{'weights', '("fractions": \{[^}]*"w_hh": )15', '$1 -1', 'fractions.w_hh must be a whole number from 0 to 62'}}, ...
            'output_fraction_63', {{'weights', '("fractions": \{[^}]*"w_out": )15', '$1 63', 'fractions.w_out must be a whole number from 0 to 62'}}, ...
            'bias_fraction_63', {{'weights', '"bias": 20', '"bias": 63', 'fractions.bias must be a whole number from 0 to 62'}}, ...
            'bias_fraction_is_not_the_pre_activation_fraction', {{'weights', '"bias": 20', '"bias": 19', 'bias fraction must equal'}}, ...
            'input_fraction_63', {{'spec', '("x": \{\s*"bits": 16,\s*"frac": )14', '$1 63', 'x.frac must be a whole number from 0 to 62'}}, ...
            'input_fraction_half', {{'spec', '("x": \{\s*"bits": 16,\s*"frac": )14', '$1 14.5', 'x.frac must be a whole number'}}, ...
            'state_fraction_63', {{'spec', '("h": \{\s*"bits": 16,\s*"frac": )15', '$1 63', 'h.frac must be a whole number from 0 to 62'}}, ...
            'output_format_fraction_63', {{'spec', '("y": \{\s*"bits": 16,\s*"frac": )14', '$1 63', 'y.frac must be a whole number from 0 to 62'}}, ...
            'pre_fraction_63', {{'spec', '("pre": \{\s*"bits": 32,\s*"frac": )20', '$1 63', 'pre.frac must be a whole number from 0 to 62'}}, ...
            'input_bits_17', {{'spec', '("x": \{\s*"bits": )16', '$1 17', 'x.bits must be a whole number from 2 to 16'}}, ...
            'input_bits_1', {{'spec', '("x": \{\s*"bits": )16', '$1 1', 'x.bits must be a whole number from 2 to 16'}}, ...
            'state_bits_17', {{'spec', '("h": \{\s*"bits": )16', '$1 17', 'h.bits must be a whole number from 2 to 16'}}, ...
            'output_bits_17', {{'spec', '("y": \{\s*"bits": )16', '$1 17', 'y.bits must be a whole number from 2 to 16'}}, ...
            'pre_bits_33', {{'spec', '("pre": \{\s*"bits": )32', '$1 33', 'pre.bits must be a whole number from 2 to 32'}}, ...
            'weight_bits_17', {{'spec', '"weight_bits": 16', '"weight_bits": 17', 'weight_bits must be a whole number from 4 to 16'}}, ...
            'weight_bits_3', {{'spec', '"weight_bits": 16', '"weight_bits": 3', 'weight_bits must be a whole number from 4 to 16'}}, ...
            'accumulator_bits_54', {{'spec', '"accumulator_bits": 48', '"accumulator_bits": 54', 'accumulator_bits must be a whole number from 32 to 53'}}, ...
            'accumulator_bits_31', {{'spec', '"accumulator_bits": 48', '"accumulator_bits": 31', 'accumulator_bits must be a whole number from 32 to 53'}}, ...
            'index_fraction_17', {{'spec', '("sigmoid": \{[^}]*"index_frac": )8', '$1 17', 'sigmoid.index_frac must be a whole number from 0 to 16'}}, ...
            'tanh_index_fraction_negative', {{'spec', '("tanh": \{[^}]*"index_frac": )8', '$1 -1', 'tanh.index_frac must be a whole number from 0 to 16'}}, ...
            'table_value_bits_17', {{'spec', '("sigmoid": \{.*?"value": \{\s*"bits": )16', '$1 17', 'sigmoid.value.bits must be a whole number from 2 to 16'}}, ...
            'table_range_is_text', {{'spec', '"range": 8.0', '"range": "8"', 'range must be a finite number'}}, ...
            'table_range_is_nan', {{'spec', '"range": 8.0', '"range": NaN', 'range must be a finite number'}}, ...
            'table_range_is_infinite', {{'spec', '"range": 8.0', '"range": Infinity', 'range must be a finite number'}}, ...
            'weight_is_nan', {{'weights', '"w_ih": \[\[-?\d+,', '"w_ih": [[NaN,', 'w_ih must hold whole numbers'}}, ...
            'weight_is_infinite', {{'weights', '"w_hh": \[\[-?\d+,', '"w_hh": [[Infinity,', 'w_hh must hold whole numbers'}}, ...
            'weight_is_half', {{'weights', '"w_out": \[\[-?\d+,', '"w_out": [[0.5,', 'w_out must hold whole numbers'}}, ...
            'weight_beyond_its_word', {{'weights', '"w_ih": \[\[-?\d+,', '"w_ih": [[32768,', 'w_ih has values outside'}}, ...
            'weight_below_its_word', {{'weights', '"w_ih": \[\[-?\d+,', '"w_ih": [[-32769,', 'w_ih has values outside'}}, ...
            'bias_beyond_the_pre_activation_word', {{'weights', '"b_out": \[-?\d+,', '"b_out": [2147483648,', 'b_out has values outside'}}, ...
            'bias_below_the_pre_activation_word', {{'weights', '"b_out": \[-?\d+,', '"b_out": [-2147483649,', 'b_out has values outside'}}, ...
            'table_value_beyond_its_word', {{'weights', '("sigmoid_table": \[)-?\d+,', '$1 40000,', 'sigmoid_table has values outside'}}, ...
            'table_value_below_its_word', {{'weights', '("sigmoid_table": \[)-?\d+,', '$1 -40000,', 'sigmoid_table has values outside'}}, ...
            'hidden_is_not_whole', {{'weights', '"hidden": 6', '"hidden": 6.5', 'hidden must be a whole number'}}, ...
            'hidden_is_zero', {{'weights', '"hidden": 6', '"hidden": 0', 'hidden must be a whole number from 1 to 512'}}, ...
            'input_count_is_three', {{'weights', '"inputs": 2', '"inputs": 3', 'inputs and outputs must both be 2'}}, ...
            'output_count_is_a_character', {{'weights', '"outputs": 2', '"outputs": "\\u0002"', 'inputs and outputs must both be 2'}}, ...
            'output_count_is_three', {{'weights', '"outputs": 2', '"outputs": 3', 'inputs and outputs must both be 2'}}, ...
            'gate_order_has_a_number', {{'weights', '"gate_order": \[[^\]]*\]', '"gate_order": [114, "z", "n"]', 'order other than r, z, n'}}, ...
            'gate_order_is_reversed', {{'weights', '"gate_order": \[[^\]]*\]', '"gate_order": ["n", "z", "r"]', 'order other than r, z, n'}}, ...
            'gate_order_is_short', {{'weights', '"gate_order": \[[^\]]*\]', '"gate_order": ["r", "z"]', 'order other than r, z, n'}}, ...
            'specification_id_of_the_weights_differs', {{'weights', '"spec_id": "fixed-point-v1"', '"spec_id": "fixed-point-v2"', 'not for fixed-point-v1'}}, ...
            'model_key_differs', {{'spec', '"model_key": "gru_stream"', '"model_key": "gru_other"', 'implements gru_stream'}}, ...
            'rounding_rule_differs', {{'spec', '"rounding": "round half up', '"rounding": "round half down', 'rules that differ'}}, ...
            'saturation_rule_differs', {{'spec', '"saturation": "every', '"saturation": "some', 'rules that differ'}}, ...
            'nonlinearity_rule_differs', {{'spec', '"nonlinearity": "table lookup', '"nonlinearity": "interpolated lookup', 'rules that differ'}}, ...
            'sigmoid_declared_as_tanh', {{'spec', '"function": "sigmoid"', '"function": "tanh"', 'is declared as'}}, ...
            'tanh_declared_as_sigmoid', {{'spec', '"function": "tanh"', '"function": "sigmoid"', 'is declared as'}}, ...
            'hidden_does_not_match_the_manifest', {{'weights', '"hidden": 6', '"hidden": 5', 'The manifest says 6 hidden units and weights.json says 5'}})
        % text the format defines, wrapped in a list: strcmp would take {'text'} for 'text'
        wrappedText = struct( ...
            'specification_id', {{'spec', '("spec_id": )("fixed-point-v1")', '$1[$2]'}}, ...
            'rounding_rule', {{'spec', '("rounding": )("[^"]*")', '$1[$2]'}}, ...
            'saturation_rule', {{'spec', '("saturation": )("[^"]*")', '$1[$2]'}}, ...
            'nonlinearity_rule', {{'spec', '("nonlinearity": )("[^"]*")', '$1[$2]'}}, ...
            'model_key_of_the_specification', {{'spec', '("model_key": )("gru_stream")', '$1[$2]'}}, ...
            'function_of_a_table', {{'spec', '("function": )("sigmoid")', '$1[$2]'}}, ...
            'specification_id_of_the_weights', {{'weights', '("spec_id": )("fixed-point-v1")', '$1[$2]'}}, ...
            'fractions_of_the_weights', {{'weights', '("fractions": )(\{[^}]*\})', '$1[$2,$2]'}}, ...
            'manifest_files_as_a_list', {{'manifest', '("files": )(\{[^}]*\})', '$1[$2,$2]'}}, ...
            'manifest_verification_as_a_list', {{'manifest', '("verification": )(\{[^}]*\})', '$1[$2,$2]'}})
        % members of the specification and of the weights that the format does define, taken out or added
        changedMembers = struct( ...
            'specification_lacks_a_rule', {{'spec', '\s*"saturation": "[^"]*",', ''}}, ...
            'specification_has_an_extra_member', {{'spec', '("spec_id": "fixed-point-v1",)', '$1 "extra": 1,'}}, ...
            'word_format_lacks_its_bits', {{'spec', '("x": \{\s*)"bits": 16,\s*', '$1'}}, ...
            'word_format_has_an_extra_member', {{'spec', '("x": \{\s*"bits": 16,)', '$1 "extra": 1,'}}, ...
            'table_lacks_its_index_fraction', {{'spec', '\s*"index_frac": 8,', ''}}, ...
            'table_has_an_extra_member', {{'spec', '("function": "sigmoid",)', '$1 "extra": 1,'}}, ...
            'weights_lack_the_input_count', {{'weights', '"inputs": 2, ', ''}}, ...
            'weights_have_an_extra_member', {{'weights', '("hidden": 6,)', '$1 "extra": 1,'}}, ...
            'fractions_lack_one', {{'weights', '"w_out": 15, ("bias")', '$1'}}, ...
            'fractions_have_an_extra_member', {{'weights', '("fractions": \{)', '$1"extra": 1, '}})
        % an accumulator is shifted right by the sum of its weight and operand fractions less the pre-activation's (20 here);
        % the references hold at most 62 places. The state fraction is raised to 21 so that 62 + 21 - 20 = 63 places are needed.
        farShift = struct('recurrent', 'w_hh', 'output', 'w_out')
        goldenHash = struct('input', 'input_sha256', 'output', 'output_sha256', 'state', 'state_sha256', ...
            'trace', 'trace_sha256')
        badWeights = struct( ...
            'other_gate_order', {{'"gate_order": \[\s*"r",\s*"z",\s*"n"\s*\]', '"gate_order": ["z", "r", "n"]'}}, ...
            'three_inputs', {{'"inputs": 2', '"inputs": 3'}}, ...
            'bias_fraction', {{'"bias": 20', '"bias": 19'}}, ...
            'weight_fraction_out_of_range', {{'"w_ih": 15,', '"w_ih": 63,'}}, ...
            'weights_for_another_specification', {{'"spec_id": "fixed-point-v1"', '"spec_id": "fixed-point-v2"'}}, ...
            'hidden_size_disagrees', {{'"hidden": 6', '"hidden": 7'}}, ...
            'a_weight_that_is_not_whole', {{'("w_ih": \[\s*\[\s*)(-?\d+)', '$1$2.5'}}, ...
            'a_weight_beyond_its_word', {{'("w_hh": \[\s*\[\s*)(-?\d+)', '$1 32768'}})
    end

    methods (TestClassSetup)
        function toolboxUnderTest(testCase)
            PackageTools.assertToolboxUnderTest(testCase);
        end
    end

    methods (Test)
        % ----- the golden vectors ------------------------------------------------------------------------------------
        function goldenVectorsAreReproducedBitForBit(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            testCase.verifyClass(model, 'opendpd.FixedModel');
            report = opendpd.verify(model);
            testCase.verifyTrue(report.passed);
            testCase.verifyEqual(report.status, 'bit_exact');
            testCase.verifyEqual(report.cases_checked, 6);
            testCase.verifyEqual(report.samples_checked, sum([model.Manifest.golden.n_samples]));
            testCase.verifyEmpty(report.mismatch_case);
            testCase.verifyEmpty(report.mismatch_sample);
            testCase.verifyEqual(report.package_c99_status, 'bit_exact');
            testCase.verifyEqual(report.spec_id, 'fixed-point-v1');
        end

        function theTwoPackagesDifferInEveryFormatTheReaderMustTakeFromThePackage(testCase)
            a = opendpd.load(FixedTools.path('gru-pa')).Manifest.spec;
            b = opendpd.load(FixedTools.path('gru-pa-custom')).Manifest.spec;
            testCase.verifyNotEqual(a.x, b.x);
            testCase.verifyNotEqual(a.h, b.h);
            testCase.verifyNotEqual(a.y, b.y);
            testCase.verifyNotEqual(b.x.frac, b.y.frac);                 % scaling the output by the input's fraction would show
            testCase.verifyNotEqual(b.x.frac, b.h.frac);
            testCase.verifyNotEqual(b.y.frac, b.h.frac);
            testCase.verifyNotEqual(a.sigmoid.value, b.sigmoid.value);
            testCase.verifyNotEqual(a.pre, b.pre);
            testCase.verifyNotEqual(a.weight_bits, b.weight_bits);
            testCase.verifyNotEqual(a.accumulator_bits, b.accumulator_bits);
            testCase.verifyNotEqual(a.sigmoid.index_frac, b.sigmoid.index_frac);
            testCase.verifyNotEqual(a.tanh.index_frac, b.tanh.index_frac);
            testCase.verifyNotEqual(a.sigmoid.range, b.sigmoid.range);
            testCase.verifyNotEqual(a.tanh.range, b.tanh.range);
        end

        function runReproducesTheGoldenIntegersThroughThePublicInterface(testCase, package)
            file = FixedTools.path(package);
            model = opendpd.load(file);
            read = opendpd.internal.readFixedPackage(file);
            hidden = model.Manifest.hidden_size;
            for entry = reshape(model.Manifest.golden, 1, [])
                g = read.golden.(entry.case_id);
                n = entry.n_samples;
                resets = reshape(entry.resets_at, 1, []) + 1;
                [yq, state, trace] = model.runInteger(reshape(g.x, 2, n).', ResetAt=resets);
                testCase.verifyEqual(yq, reshape(g.y, 2, n).', "outputs of " + entry.case_id);
                testCase.verifyEqual(trace, reshape(g.h_trace, hidden, n).', "state steps of " + entry.case_id);
                testCase.verifyEqual(state, g.h_final, "final state of " + entry.case_id);
            end
        end

        function theResultDoesNotDependOnHowTheStreamIsCutIntoChunks(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            read = opendpd.internal.readFixedPackage(FixedTools.path(package));
            n = 718;
            x = reshape(read.golden.normal.x, 2, n).';
            whole = model.runInteger(x);
            stream = RandStream('twister', Seed=5);
            chunked = zeros(size(whole), 'like', whole);
            state = [];
            first = 1;
            while first <= n
                last = min(n, first + randi(stream, 90));
                [chunked(first:last, :), state] = model.runInteger(x(first:last, :), State=state);
                first = last + 1;
            end
            testCase.verifyEqual(chunked, whole);
            [~, final] = model.runInteger(x);
            testCase.verifyEqual(state, final);
            waveform = complex(double(x(:, 1)), double(x(:, 2))) / 2^model.Manifest.spec.x.frac;
            testCase.verifyEqual(model.apply(waveform, ChunkSamples=37), model.apply(waveform));
        end

        function theStateIsCarriedAndOnlyAResetClearsIt(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            read = opendpd.internal.readFixedPackage(FixedTools.path(package));
            x = reshape(read.golden.normal.x, 2, 718).';
            block = x(1:100, :);
            first = model.runInteger(block);
            carried = model.runInteger([block; block]);
            testCase.verifyEqual(carried(1:100, :), first);
            testCase.verifyNotEqual(carried(101:200, :), first, 'the second block must see the state the first one left');
            reset = model.runInteger([block; block], ResetAt=101);
            testCase.verifyEqual(reset(101:200, :), first);
            % a reset before sample 50 makes the rest of the run what a fresh run of those samples is; one before sample 1 changes nothing
            [~, ~, resumed] = model.runInteger(block, ResetAt=[1 50]);
            [~, ~, fresh] = model.runInteger(block(50:end, :));
            testCase.verifyEqual(resumed(50:end, :), fresh);
            [~, ~, plain] = model.runInteger(block);
            testCase.verifyEqual(resumed(1:49, :), plain(1:49, :));
            testCase.verifyNotEqual(resumed(50:end, :), plain(50:end, :));
            % a reset before the first sample clears a state the run was given; one before the last sample clears it there
            short = block(1:10, :);
            [~, mid] = model.runInteger(block);
            [givenState, ~, givenTrace] = model.runInteger(short, State=mid);
            zeroState = model.runInteger(short);
            testCase.verifyNotEqual(givenState, zeroState, 'the state the run is given must matter');
            testCase.verifyEqual(model.runInteger(short, State=mid, ResetAt=1), zeroState);
            [~, ~, lastReset] = model.runInteger(short, State=mid, ResetAt=10);
            [~, ~, alone] = model.runInteger(short(10, :));
            testCase.verifyEqual(lastReset(1:9, :), givenTrace(1:9, :));
            testCase.verifyEqual(lastReset(10, :), alone);
            testCase.verifyNotEqual(givenTrace(10, :), alone);
        end

        function aResetBeyondTheLastSampleIsIgnoredHoweverFarItIs(testCase)
            % ResetAt takes any positive whole number, so the kernel must not size anything by it
            model = opendpd.load(FixedTools.path('gru-pa'));
            read = opendpd.internal.readFixedPackage(FixedTools.path('gru-pa'));
            block = reshape(read.golden.normal.x, 2, 718).';
            block = block(1:100, :);
            [expected, ~, trace] = model.runInteger(block, ResetAt=50);
            [~, ~, plain] = model.runInteger(block);
            testCase.verifyNotEqual(trace(50:end, :), plain(50:end, :), 'the reset at 50 must matter');
            [actual, ~, far] = model.runInteger(block, ResetAt=[50 101 1e6 2^40 1e15]);
            testCase.verifyEqual(actual, expected);
            testCase.verifyEqual(far, trace);
        end

        % ----- the operators ------------------------------------------------------------------------------------------
        function rescaleRoundsHalfUpAndShiftsLeftExactly(testCase)
            rescale = @opendpd.runtime.fixedRescale;
            testCase.verifyEqual(rescale([5 -5 6 -6 7 -7], 2, 0), [1 -1 2 -1 2 -2]);       % /4: 1.25 -1.25 1.5 -1.5 1.75 -1.75
            testCase.verifyEqual(rescale([0 1 -1 2 -2], 1, 0), [0 1 0 1 -1]);                % /2: 0 .5 -.5 1 -1, halves go up
            testCase.verifyEqual(rescale(3, 0, 4), 48);
            testCase.verifyEqual(rescale([-3 0 3], 5, 5), [-3 0 3]);
            stream = RandStream('twister', Seed=3);
            for k = 1:2000
                % left shifts of at most 10 bits keep the int64 oracle below its own saturation (2^63)
                v = round((rand(stream) * 2 - 1) * 2^(1 + floor(rand(stream) * 46)));
                from = floor(rand(stream) * 41);
                to = max(0, from + floor(rand(stream) * 51) - 40);
                testCase.verifyEqual(rescale(v, from, to), double(FixedTools.rescale(int64(v), from, to)), ...
                    sprintf('rescale(%d, %d, %d)', v, from, to));
            end
        end

        function inputsAreQuantisedHalfAwayFromZeroAndSaturated(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));                      % 16 bits, 14 fractional
            half = 2^-15;
            testCase.verifyEqual(model.quantiseInput(complex([1 -1 2.5 0.00003 -2.5 2].', [0 0 0 0 -3 -2].')), ...
                [16384 0; -16384 0; 32767 0; 0 0; -32768 -32768; 32767 -32768]);
            testCase.verifyEqual(model.quantiseInput(complex([half -half 3 * half -3 * half 0.5 * half].', zeros(5, 1))), ...
                [1 0; -1 0; 2 0; -2 0; 0 0]);
            % the input is rounded to single first: this double is just below half an LSB, its single is exactly half
            testCase.verifyEqual(model.quantiseInput(complex(2^-15 - 1e-13, 0)), [1 0]);
            testCase.verifyEqual(model.quantiseInput(complex(-(2^-15 - 1e-13), 0)), [-1 0]);
            custom = opendpd.load(FixedTools.path('gru-pa-custom'));              % 14 bits, 12 fractional
            testCase.verifyEqual(custom.quantiseInput(complex([1 -1 2.5 -3 2^-13 -2^-13].', zeros(6, 1))), ...
                [4096 0; -4096 0; 8191 0; -8192 0; 1 0; -1 0]);
        end

        function applyScalesTheIntegerResultByTheOutputFormat(testCase, package)
            model = opendpd.load(FixedTools.path(package));
            stream = RandStream('twister', Seed=8);
            x = 0.4 * complex(randn(stream, 300, 1), randn(stream, 300, 1));
            [y, info] = opendpd.apply(model, x);
            testCase.verifyClass(y, 'single');
            testCase.verifySize(y, [300 1]);
            yq = double(model.runInteger(model.quantiseInput(x)));
            testCase.verifyEqual(double([real(y) imag(y)]), yq / 2^model.Manifest.spec.y.frac, AbsTol=0, RelTol=0);
            testCase.verifyEqual(info.execution, 'streaming_stateful');
            testCase.verifyEqual(info.spec_id, 'fixed-point-v1');
            testCase.verifyEqual(opendpd.apply(model, single(x)), y);
            testCase.verifyEqual(opendpd.apply(model, x, Execution="streaming"), y);
            testCase.verifyEqual(opendpd.apply(model, x, Execution="auto"), y);              % its one execution, so auto is that
            testCase.verifyEqual(opendpd.apply(model, x, Execution="streaming_stateful", ChunkSamples=64), y);
        end

        function aFixedModelHasNoOfflineExecution(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            testCase.verifyError(@() opendpd.apply(model, ones(10, 1), Execution="offline_segmented"), 'opendpd:NoOfflineVariant');
        end

        % ----- the kernel against an independent implementation -------------------------------------------------------
        function theKernelEqualsAnIndependentIntegerImplementationOnRandomFormats(testCase, seed)
            stream = RandStream('twister', Seed=seed);
            f = FixedTools.randomFormat(stream, seed > 12);                  % half of the seeds make every sum saturate
            n = 150;
            x = round((rand(stream, n, 2) * 2 - 1) * 2^(f.xBits - 1) * 1.3);          % beyond full scale: the input saturates
            extreme = rand(stream, n, 1) < 0.1;
            x(extreme, :) = repmat([f.xMax f.xMin], nnz(extreme), 1);
            resets = sort(randperm(stream, n, 3));
            h0 = round((rand(stream, f.hidden, 1) * 2 - 1) * f.hMax);
            [y, h, trace] = opendpd.runtime.fixedGruRun(x, h0, resets, f);
            [yr, hr, tr] = FixedTools.referenceRun(f, x, h0, resets);
            testCase.verifyEqual(y, yr);
            testCase.verifyEqual(h, hr);
            testCase.verifyEqual(trace, tr);
            testCase.verifyLessThanOrEqual(max(abs(y(:))), max(f.yMax, -f.yMin));
        end

        function anAccumulatorThatFillsItsWidthRaisesAnError(testCase)
            f = FixedTools.randomFormat(RandStream('twister', Seed=2));
            f.accBits = 20;
            f.accLimit = 2^19;
            % each of the three dot products in turn: input times weights, state times weights, new state times output weights
            inputs = f; inputs.wIH(:) = 2^14;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun(repmat(f.xMax, 4, 2), zeros(f.hidden, 1), zeros(1, 0), inputs), ...
                'opendpd:FixedOverflow');
            recurrent = f; recurrent.wHH(:) = 2^14;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun(zeros(4, 2), repmat(f.hMax, f.hidden, 1), zeros(1, 0), ...
                recurrent), 'opendpd:FixedOverflow');
            output = f; output.wOut(:) = 2^14;
            output.sigmoid.values(:) = 0;                                  % z = 0: the new state is the candidate, at its maximum
            output.tanh.values(:) = output.hMax;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun(zeros(4, 2), zeros(f.hidden, 1), zeros(1, 0), output), ...
                'opendpd:FixedOverflow');
            quiet = f; quiet.accBits = 48; quiet.accLimit = 2^47;           % the same weights fit a wide enough accumulator
            quiet.wIH(:) = 2^14;
            [y, ~, ~] = opendpd.runtime.fixedGruRun(repmat(f.xMax, 4, 2), zeros(f.hidden, 1), zeros(1, 0), quiet);
            testCase.verifySize(y, [4 2]);
        end

        function theLimitOfAnAccumulatorIsAnOverflowAndOneLessIsNot(testCase)
            f = FixedTools.randomFormat(RandStream('twister', Seed=6));
            f.accBits = 20;
            f.accLimit = 2^19;
            f.xMin = -2^15; f.xMax = 2^15 - 1; f.hMin = -2^15; f.hMax = 2^15 - 1;
            f.wIH(:) = 0; f.wHH(:) = 0; f.wOut(:) = 0;                      % each case below switches on one dot product only
            none = zeros(1, 0);
            quiet = zeros(f.hidden, 1);
            input = f; input.wIH(1, :) = [2^9 0];
            testCase.verifyError(@() opendpd.runtime.fixedGruRun([2^10 0], quiet, none, input), 'opendpd:FixedOverflow');
            testCase.verifySize(opendpd.runtime.fixedGruRun([2^10 - 1 0], quiet, none, input), [1 2]);
            recurrent = f; recurrent.wHH(1, 1) = 2^9;
            state = quiet; state(1) = 2^10;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun([0 0], state, none, recurrent), 'opendpd:FixedOverflow');
            state(1) = 2^10 - 1;
            testCase.verifySize(opendpd.runtime.fixedGruRun([0 0], state, none, recurrent), [1 2]);
            output = f; output.wOut(1, 1) = 2^11;
            output.sigmoid.values(:) = 0;                                  % z = 0: the new state is the candidate itself
            output.tanh.values(:) = 2^8;
            testCase.verifyError(@() opendpd.runtime.fixedGruRun([0 0], quiet, none, output), 'opendpd:FixedOverflow');
            output.tanh.values(:) = 2^8 - 1;
            testCase.verifySize(opendpd.runtime.fixedGruRun([0 0], quiet, none, output), [1 2]);
        end

        function theOutputPreActivationSaturatesBeforeItIsRescaledToTheOutputFormat(testCase)
            % The output format reaches +-2048 and the pre-activation only +-2: a sum that is not saturated first gives a
            % larger output than the specification does.
            f = FixedTools.randomFormat(RandStream('twister', Seed=4), true);
            f.yBits = 16; f.yFrac = 4; f.yMin = -2^15; f.yMax = 2^15 - 1;
            f.hMin = -2^15; f.hMax = 2^15 - 1; f.hFrac = 8; f.fOut = 4;
            f.sigmoid.values(:) = 0;
            f.tanh.values(:) = 100;                                         % the new state is 100 in every unit
            f.wOut = abs(f.wOut) + 1;
            f.bOut(:) = f.preMax;
            x = zeros(5, 2);
            y = opendpd.runtime.fixedGruRun(x, zeros(f.hidden, 1), zeros(1, 0), f);
            testCase.verifyEqual(y, 32 * ones(5, 2), 'floor((2^23 - 1 + 2^17) / 2^18) = 32');
            testCase.verifyEqual(y, FixedTools.referenceRun(f, x, zeros(f.hidden, 1), zeros(1, 0)));
        end

        % ----- verification finds the fault ----------------------------------------------------------------------------
        function verifyLocatesTheFirstDifferingSampleAndSignal(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            outputOnly = FixedTools.editGolden(parts, "state_reset", "y.i16", 2 * 99 + 1, 1);       % sample 100, I of the output
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, outputOnly)));
            testCase.verifyFalse(report.passed);
            testCase.verifyEqual(report.status, 'mismatch');
            testCase.verifyEqual(report.mismatch_case, 'state_reset');
            testCase.verifyEqual(report.mismatch_sample, 100);
            testCase.verifyEqual(report.mismatch_signal, 'y');
            testCase.verifyEqual(report.cases_checked, 4);
            stateOnly = FixedTools.editGolden(parts, "extreme", "h_trace.i16", 6 * 49 + 3, -1);     % sample 50, unit 3 of the state
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, stateOnly)));
            testCase.verifyEqual([string(report.mismatch_case) string(report.mismatch_signal)], ["extreme" "h"]);
            testCase.verifyEqual(report.mismatch_sample, 50);
            testCase.verifyEqual(report.cases_checked, 1);
            % both differ at one sample: the state comes first, because the output is computed from it
            both = FixedTools.editGolden(stateOnly, "extreme", "y.i16", 2 * 49 + 2, 1);
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, both)));
            testCase.verifyEqual(report.mismatch_signal, 'h');
            testCase.verifyEqual(report.mismatch_sample, 50);
            % a state that differs later than an output is reported at the output
            later = FixedTools.editGolden(parts, "saturation", "h_trace.i16", 6 * 200 + 1, 1);
            later = FixedTools.editGolden(later, "saturation", "y.i16", 2 * 30 + 1, 1);
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, later)));
            testCase.verifyEqual([string(report.mismatch_signal) string(report.mismatch_case)], ["y" "saturation"]);
            testCase.verifyEqual(report.mismatch_sample, 31);
        end

        % ----- the archive ---------------------------------------------------------------------------------------------
        function unexpectedEntriesAreRefusedAndNeverExtracted(testCase)
            parts = PackageTools.withEntry(FixedTools.unpack(FixedTools.path('gru-pa')), "evil.m", uint8('disp(''pwned'')'));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageContents');
            testCase.verifyEmpty(which('evil'));
        end

        function pathsOutsideThePackageAreRefused(testCase, badPath)
            escape = fullfile(tempdir, 'opendpd_escape.txt');
            PackageTools.deleteIfPresent(escape);
            parts = PackageTools.withEntry(FixedTools.unpack(FixedTools.path('gru-pa')), badPath, uint8('x'));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageContents');
            testCase.verifyFalse(isfile(escape));
            testCase.verifyFalse(isfile('/tmp/opendpd_escape.txt'));
        end

        function aModifiedFileIsRefused(testCase, modified)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            k = find(parts.names == modified);
            parts.bytes{k}(end) = bitxor(parts.bytes{k}(end), uint8(1));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageHash');
        end

        function aFileTheManifestDoesNotListIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editManifest(parts, '\s*"c/harness.c": "[0-9a-f]{64}",', '');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:PackageHash');
        end

        function aFileTheManifestListsButTheArchiveLacksIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, PackageTools.withoutEntry(parts, "c/harness.c"))), ...
                'opendpd:PackageHash');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, PackageTools.withoutEntry(parts, "weights.json"))), ...
                'opendpd:PackageContents');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, PackageTools.withoutEntry(parts, "manifest.json"))), ...
                'opendpd:PackageContents');
        end

        function duplicateEntriesAreRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            twin = parts.bytes{parts.names == "c/gru_fixed.h"};
            parts = PackageTools.withEntry(parts, "c/gru_fixed.x", twin);
            file = FixedTools.write(testCase, parts);
            PackageTools.writeBytes(file, PackageTools.renameInZip(PackageTools.readBytes(file), "c/gru_fixed.x", "c/gru_fixed.h"));
            testCase.verifySubstring(refusalOf(@() opendpd.load(file)), 'appears more than once');
            try
                opendpd.load(file);
            catch cause
                testCase.verifyEqual(cause.identifier, 'opendpd:PackageContents');
            end
        end

        function anEntryThatInflatesBeyondItsDeclaredSizeIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.replaceBytes(parts, "golden/normal/meta.json", zeros(3e6, 1, 'uint8'));
            file = FixedTools.write(testCase, parts);
            testCase.assertLessThan(dir(file).bytes, 1e6, 'the test archive should be small');
            lying = fullfile(fileparts(file), 'lying.fixed-point-v1.zip');
            PackageTools.writeBytes(lying, PackageTools.patchDeclaredSize(PackageTools.readBytes(file), ...
                'golden/normal/meta.json', 1000));
            testCase.verifyError(@() opendpd.load(lying), 'opendpd:PackageContents');
        end

        function sizeLimitsAreEnforced(testCase)
            file = FixedTools.path('gru-pa');
            testCase.verifyError(@() opendpd.internal.readFixedPackage(file, MaxEntryBytes=20000), 'opendpd:PackageContents');
            testCase.verifyError(@() opendpd.internal.readFixedPackage(file, MaxTotalBytes=50000), 'opendpd:PackageContents');
            package = opendpd.internal.readFixedPackage(file);
            testCase.verifyEqual(package.manifest.spec.spec_id, 'fixed-point-v1');
        end

        function aFileThatIsNotAPackageIsRefused(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            text = fullfile(folder, 'not-a-zip.zip');
            PackageTools.writeBytes(text, uint8('this is not a zip file'));
            testCase.verifyError(@() opendpd.load(text), 'opendpd:Package');
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            for edit = {'"fixed-point-v1"', '"fixed-point-v2"'; '^.*$', 'not json'}.'
                changed = FixedTools.editManifest(parts, edit{1}, edit{2});
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, changed)), 'opendpd:Package');
            end
            json = FixedTools.replaceBytes(parts, "weights.json", uint8('{"hidden": '));
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, json)), 'opendpd:Package');
        end

        function onlyTheSixCasesOfTheSpecificationAreAccepted(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            for name = ["golden/1abc/x.i16", "golden/a-b/x.i16", "golden/extra/x.i16", "golden/Normal/x.i16", ...
                    "golden/normal2/x.i16", "golden/normal/x.i16.bak", "golden/normal/sub/x.i16", "golden/x.i16"]
                extra = PackageTools.withEntry(parts, name, uint8([1 0]));
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, extra)), 'opendpd:PackageContents', char(name));
            end
            % a case that is listed everywhere, in the archive and in the manifest, under a name that is not one of the six
            renamed = parts;
            renamed.names = replace(renamed.names, "golden/normal/", "golden/end/");
            renamed = FixedTools.editManifest(renamed, '"golden/normal/', '"golden/end/');
            renamed = FixedTools.editManifest(renamed, '"case_id": "normal"', '"case_id": "end"');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, renamed)), 'opendpd:PackageContents');
        end

        function aDirectoryEntryTheFormatDoesNotDefineIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            for name = ["golden/extra/", "extra/", "c/sub/", "golden/normal/sub/"]
                extra = PackageTools.withEntry(parts, name, uint8([]));
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, extra)), 'opendpd:PackageContents', char(name));
            end
        end

        % ----- what the package says --------------------------------------------------------------------------------------
        function aSpecificationThisToolboxDoesNotImplementIsRefused(testCase, badSpec)
            parts = FixedTools.editSpec(FixedTools.unpack(FixedTools.path('gru-pa')), badSpec{1}, badSpec{2});
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aSpecificationThatDiffersFromTheManifestsIsRefused(testCase)
            % a difference nothing else would notice: a 47-bit accumulator loads fine on its own
            parts = FixedTools.editJson(FixedTools.unpack(FixedTools.path('gru-pa')), "spec.json", ...
                '"accumulator_bits": 48', '"accumulator_bits": 47');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
            alone = FixedTools.editSpec(FixedTools.unpack(FixedTools.path('gru-pa')), '"accumulator_bits": 48', '"accumulator_bits": 47');
            testCase.verifyEqual(opendpd.verify(opendpd.load(FixedTools.write(testCase, alone))).status, 'bit_exact');
        end

        function aGoldenFileThatIsShorterThanItsIndexSaysIsRefused(testCase, shortGolden)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            name = "golden/all_zero/" + shortGolden;
            bytes = parts.bytes{parts.names == name};
            short = FixedTools.replaceBytes(parts, name, bytes(1:end - 2));
            testCase.verifySubstring(refusalMessage(testCase, short, 'opendpd:Package'), 'sizes its index declares');
            long = FixedTools.replaceBytes(parts, name, [bytes(:); uint8([0; 0])]);
            testCase.verifySubstring(refusalMessage(testCase, long, 'opendpd:Package'), 'sizes its index declares');
        end

        function manifestFieldsThatAreShownOrReportedAreChecked(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            runId = regexp(FixedTools.textOf(parts, "manifest.json"), '"run_id": "([^"]*)"', 'tokens', 'once');
            pattern = ['"run_id": "' regexptranslate('escape', runId{1}) '"'];
            for unsafe = ["has space", "esc\\u001b[2Jx", "../up", "new\\nline"]
                changed = FixedTools.editManifest(parts, pattern, ['"run_id": "' char(unsafe) '"']);
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, changed)), 'opendpd:Package', char(unsafe));
            end
            verdict = FixedTools.editManifest(parts, '"status": "bit_exact"', '"status": "perfect"');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, verdict)), 'opendpd:Package');
            size = FixedTools.editManifest(parts, '"hidden_size": 6', '"hidden_size": 7');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, size)), 'opendpd:Package');
        end

        function weightsThatAreNotWhatTheSpecificationDescribesAreRefused(testCase, badWeights)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editJson(parts, "weights.json", badWeights{1}, badWeights{2});
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aPackageWhoseIntegersCouldLeaveTheExactRangeOfDoublePrecisionIsRefused(testCase)
            % Weights and inputs with no fractional bits, scaled up by 20 bits to the pre-activation format: a 48-bit
            % accumulator could then reach 2^67. Everything else in the package is consistent, so only the bound can refuse it.
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editJson(parts, "weights.json", '"w_ih": 15,', '"w_ih": 0,');
            parts = FixedTools.editManifest(parts, ...
                '("name": "w_ih",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )15', '$1 0');
            parts = FixedTools.editSpec(parts, '("x": \{\s*"bits": 16,\s*"frac": )14', '$1 0');
            file = FixedTools.write(testCase, parts);
            try
                opendpd.load(file);
                message = '';
            catch cause
                testCase.verifyEqual(cause.identifier, 'opendpd:Package');
                message = cause.message;
            end
            testCase.verifySubstring(message, 'exactly');
        end

        function aManifestWithoutTheFileHashesIsNotAPackage(testCase)
            parts = FixedTools.editManifest(FixedTools.unpack(FixedTools.path('gru-pa')), '"files":', '"file_hashes":');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aTableOfTheWrongLengthIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editJson(parts, "weights.json", '("tanh_table": \[\s*)(-?\d+),', '$1');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aManifestThatDisagreesAboutTheQuantisationIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            parts = FixedTools.editManifest(parts, '("name": "w_hh",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )15', '$1 14');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function goldenVectorsThatDoNotMatchTheirIndexAreRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            lie = FixedTools.editManifest(parts, '("case_id": "all_zero",\s*"description": "[^"]*",\s*"n_samples": )256', '$1 257');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, lie)), 'opendpd:Package');
            samples = regexp(FixedTools.textOf(parts, "manifest.json"), '"case_id": "state_reset",\s*"description": "[^"]*",\s*"n_samples": (\d+)', 'tokens', 'once');
            for resets = ["5000", samples{1}, "-1", "256", "768.5", "1e30"]       % beyond the vector, just beyond it, below it, twice, between samples, absurd
                reset = FixedTools.editManifest(parts, '("resets_at": \[\s*0,\s*256,\s*512,\s*)768', ['$1 ' char(resets)]);
                testCase.verifySubstring(refusalMessage(testCase, reset, 'opendpd:Package'), 'outside the vector', char(resets));
            end
            missing = FixedTools.editManifest(parts, '"case_id": "long_sequence"', '"case_id": "longer_sequence"');
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, missing)), 'opendpd:Package');
            torn = FixedTools.editGolden(parts, "normal", "h_final.i16", 1, 1);
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, torn)), 'opendpd:Package');
        end

        function aGoldenFileTheIndexNamesMayNotBeMissingEvenWhenTheManifestDoesNotListIt(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            for file = ["x.i16", "y.i16", "h_final.i16", "h_trace.i16"]
                name = "golden/normal/" + file;
                changed = FixedTools.editManifest(parts, ['\s*"' regexptranslate('escape', char(name)) '": "[0-9a-f]{64}",?'], '');
                changed = PackageTools.withoutEntry(changed, name);
                testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:PackageContents'), char(file), char(name));
            end
        end

        function theLimitsAreTheOnesTheDocumentationStates(testCase)
            limits = opendpd.internal.fixedLimits();
            testCase.verifyEqual([limits.MaxHidden limits.MaxTableEntries limits.MaxGoldenSamples limits.MaxNesting], ...
                [512 65536 2097152 8]);
            testCase.verifyEqual([limits.ManifestBytes limits.SpecBytes limits.WeightsBytes limits.SourceBytes limits.MetaBytes ...
                limits.TraceBytes], [4e6 1e6 32e6 32e6 1e6 160e6]);
            testCase.verifyEqual([limits.ManifestSeparators limits.SpecSeparators], [4000 500]);
        end

        % ----- what a hostile package can ask of MATLAB -------------------------------------------------------------------------
        function jsonThatNestsDeeplyIsRefusedBeforeJsondecodeCanCrashMatlab(testCase)
            % jsondecode recurses once per level: 100000 levels end the MATLAB process with a segmentation fault.
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            limits = opendpd.internal.fixedLimits();
            for depth = [limits.MaxNesting + 1, 5000, 100000]
                deep = uint8([repmat('[', 1, depth) repmat(']', 1, depth)]).';
                for name = ["manifest.json", "spec.json", "weights.json"]
                    changed = FixedTools.replaceBytes(parts, name, deep);
                    message = refusalMessage(testCase, changed, 'opendpd:Package');
                    testCase.verifySubstring(message, sprintf('%s nests %d levels deep', name, depth));
                end
            end
            % the same nesting inside a member of the manifest and of the specification, which two readers once compared
            nested = ['"deep":' repmat('{"a":', 1, 8000) '1' repmat('}', 1, 8000)];
            changed = FixedTools.editSpec(parts, '"function": "sigmoid",', ['"function": "sigmoid", ' nested ',']);
            testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:Package'), 'levels deep');
            % exactly as deep as the limit allows is not a reason to refuse: the package is refused for its member, not its depth
            allowed = FixedTools.editSpec(parts, '"function": "sigmoid",', ...
                ['"function": "sigmoid", "deep":' repmat('{"a":', 1, limits.MaxNesting - 3) '1' repmat('}', 1, limits.MaxNesting - 3) ',']);
            testCase.verifySubstring(refusalMessage(testCase, allowed, 'opendpd:Package'), 'does not define');
            testCase.verifyEqual(class(opendpd.load(FixedTools.path('gru-pa'))), 'opendpd.FixedModel');
        end

        function jsonWithMoreElementsThanTheFormatCanHoldIsRefusedBeforeItIsParsed(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            junk = ['"junk": [' repmat('0,', 1, 300000) '0]'];            % 300000 numbers: jsondecode would need about 100 MB
            changed = FixedTools.editJson(parts, "weights.json", '"hidden": 6,', ['"hidden": 6, ' junk ',']);
            message = refusalMessage(testCase, changed, 'opendpd:Package');
            testCase.verifySubstring(message, 'weights.json holds');
            testCase.verifySubstring(message, 'array elements');
            many = ['"x": [' repmat('0,', 1, 5000) '0],'];
            changed = FixedTools.editManifest(parts, '"schema_version": 1,', ['"schema_version": 1, ' many]);
            testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:Package'), 'manifest.json holds');
            some = ['"x": [' repmat('0,', 1, 600) '0],'];                 % the specification gets a budget of 500
            changed = FixedTools.editSpec(parts, '"spec_id": "fixed-point-v1",', ['"spec_id": "fixed-point-v1", ' some]);
            testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:Package'), 'spec.json holds');
        end

        function theBoundsOfDecodeJsonAreInclusive(testCase)
            five = uint8('{"a":{"a":{"a":{"a":{"a":1}}}}}').';
            value = opendpd.internal.decodeJson(five, 'five', MaxDepth=5);
            testCase.verifyEqual(value.a.a.a.a.a, 1);
            testCase.verifyError(@() opendpd.internal.decodeJson(five, 'five', MaxDepth=4), 'opendpd:Package');
            nine = uint8('[0,0,0,0,0,0,0,0,0,0]').';                                       % nine commas
            testCase.verifyEqual(opendpd.internal.decodeJson(nine, 'nine', MaxSeparators=9), (zeros(10, 1)));
            testCase.verifyError(@() opendpd.internal.decodeJson(nine, 'nine', MaxSeparators=8), 'opendpd:Package');
            quoted = uint8('["a,b,c,{[", 1]').';                                            % commas and brackets in a string count for nothing
            testCase.verifyEqual(opendpd.internal.decodeJson(quoted, 'quoted', MaxDepth=1, MaxSeparators=1), {'a,b,c,{['; 1});
            testCase.verifyError(@() opendpd.internal.decodeJson(uint8('{"a": ').', 'broken'), 'opendpd:Package');
        end

        function filesLargerThanTheirKindAllowsAreRefusedBeforeTheyAreRead(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            limits = opendpd.internal.fixedLimits();
            for entry = {"weights.json", limits.WeightsBytes; "manifest.json", limits.ManifestBytes; ...
                    "spec.json", limits.SpecBytes; "golden/normal/meta.json", limits.MetaBytes; ...
                    "README.md", limits.SourceBytes; "c/gru_fixed.c", limits.SourceBytes; ...
                    "golden/normal/h_final.i16", 2 * limits.MaxHidden; "golden/normal/x.i16", 4 * limits.MaxGoldenSamples; ...
                    "golden/normal/y.i16", 4 * limits.MaxGoldenSamples; "golden/long_sequence/h_trace.i16", limits.TraceBytes}.'
                changed = FixedTools.replaceBytes(parts, entry{1}, zeros(entry{2} + 1, 1, 'uint8'));
                message = refusalMessage(testCase, changed, 'opendpd:PackageContents');
                testCase.verifySubstring(message, sprintf('the limit for this file is %g', entry{2}), char(entry{1}));
            end
        end

        function theGoldenVectorsTogetherCannotHoldMoreSamplesThanTheLimit(testCase)
            % each file is within its own limit; two of them together hold more samples than any package needs
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            limits = opendpd.internal.fixedLimits();
            half = zeros(4 * (limits.MaxGoldenSamples / 2 + 1), 1, 'uint8');
            changed = FixedTools.replaceBytes(parts, "golden/normal/x.i16", half);
            changed = FixedTools.replaceBytes(changed, "golden/extreme/x.i16", half);
            testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:PackageContents'), 'more than');
        end

        function aGoldenFileThatEndsInHalfAValueIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            bytes = parts.bytes{parts.names == "golden/all_zero/x.i16"};
            changed = FixedTools.replaceBytes(parts, "golden/all_zero/x.i16", [bytes(:); uint8(7)]);
            try
                opendpd.load(FixedTools.write(testCase, changed));
                message = '';
            catch cause
                testCase.verifyEqual(cause.identifier, 'opendpd:Package');
                message = cause.message;
            end
            testCase.verifySubstring(message, 'whole 16-bit');
        end

        function valuesOfTheWrongTypeAreRefused(testCase, typeConfusion)
            parts = editPackage(FixedTools.unpack(FixedTools.path('gru-pa')), typeConfusion);
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aRowOfCharacterCodesThatSpellsTheRightTextIsStillNotText(testCase)
            % jsondecode turns [[102,105,...]] into a 1-by-N matrix, and isequal takes that for the char row it spells
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            spec = jsondecode(FixedTools.textOf(parts, "spec.json"));
            weights = jsondecode(FixedTools.textOf(parts, "weights.json"));
            codes = @(text) ['[[' char(strjoin(string(double(char(text))), ',')) ']]'];
            testCase.verifyEqual(jsondecode(codes('ab')), [97 98]);                   % the shape that fools isequal
            for member = ["model_key", "rounding", "saturation", "nonlinearity"]
                pattern = sprintf('("%s": )("%s")', member, regexptranslate('escape', spec.(member)));    % not the manifest's own model_key
                changed = FixedTools.editSpec(parts, pattern, ['$1' codes(spec.(member))]);
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, changed)), 'opendpd:Package', char(member));
            end
            changed = FixedTools.editSpec(parts, '("function": )("sigmoid")', ['$1' codes('sigmoid')]);
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, changed)), 'opendpd:Package', 'table function');
            changed = FixedTools.editJson(parts, "weights.json", '("spec_id": )("fixed-point-v1")', ['$1' codes(weights.spec_id)]);
            testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:Package'), 'not for fixed-point-v1');
        end

        function numbersOutsideTheirRangeAreRefusedAndSaySo(testCase, outOfRange)
            parts = editPackage(FixedTools.unpack(FixedTools.path('gru-pa')), outOfRange);
            testCase.verifySubstring(refusalMessage(testCase, parts, 'opendpd:Package'), outOfRange{4});
        end

        function textInAListIsNotTextEvenWhenStrcmpWouldAcceptIt(testCase, wrappedText)
            parts = editPackage(FixedTools.unpack(FixedTools.path('gru-pa')), wrappedText);
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aMemberTheFormatDefinesCannotBeMissingAndOneItDoesNotCannotBeAdded(testCase, changedMembers)
            parts = editPackage(FixedTools.unpack(FixedTools.path('gru-pa')), changedMembers);
            message = refusalMessage(testCase, parts, 'opendpd:Package');
            testCase.verifyTrue(contains(message, 'does not define') || contains(message, 'lacks'), message);
        end

        function aManifestThatIsAListOfManifestsIsRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            text = FixedTools.textOf(parts, "manifest.json");
            both = FixedTools.replaceBytes(parts, "manifest.json", uint8(['[' text ',' text ']']).');
            testCase.verifySubstring(refusalMessage(testCase, both, 'opendpd:Package'), 'Not a fixed-point-v1 package');
            empty = FixedTools.replaceBytes(parts, "manifest.json", uint8('[]').');
            testCase.verifySubstring(refusalMessage(testCase, empty, 'opendpd:Package'), 'Not a fixed-point-v1 package');
        end

        function theChecksumOfTheCheckpointMayBeLeftOut(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            left = FixedTools.editManifest(parts, '\s*"weights_sha256": [^,]*,', '');
            testCase.verifyEqual(opendpd.verify(opendpd.load(FixedTools.write(testCase, left))).status, 'bit_exact');
        end

        function theElementBudgetOfTheWeightsIsExactlyWhatTheHiddenSizeAndTheTablesNeed(testCase)
            % hidden units: 3h x 2 input weights, 3h x h and 2 x h recurrent and output weights, 3h + 3h + 2 biases and 3 gate
            % names make 3h^2 + 14h + 2 numbers, and each table may hold the most entries the toolbox accepts
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            limits = opendpd.internal.fixedLimits();
            hidden = 6;
            budget = 3 * hidden^2 + 14 * hidden + 2 + 2 * limits.MaxTableEntries + 1000;
            text = FixedTools.textOf(parts, "weights.json");
            [~, own] = opendpd.internal.jsonShape(uint8(text(:)));
            room = budget - own;                                % commas that may still be added
            for count = [room, room + 1]
                junk = ['"junk": [' repmat('0,', 1, count - 1) '0], '];       % COUNT elements and the comma after the member
                changed = FixedTools.editJson(parts, "weights.json", '"hidden": 6,', ['"hidden": 6, ' junk]);
                message = refusalMessage(testCase, changed, 'opendpd:Package');
                testCase.verifyEqual(contains(message, 'array elements'), count > room, sprintf('%d commas added, %d allowed', count, room));
            end
        end

        function aJsonErrorMessageNeverCarriesControlCharacters(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            for text = {['{"hidden": ' char(27) '[31mred'], ['{"hidden" ' char(7) char(13) '}'], ['[1,2' char(27) ']']}
                changed = FixedTools.replaceBytes(parts, "weights.json", uint8(text{1}).');
                message = refusalMessage(testCase, changed, 'opendpd:Package');
                testCase.verifySubstring(message, 'weights.json is not valid JSON');
                testCase.verifyFalse(any(message < 32 & message ~= 10), 'the message has a control character');
            end
        end

        function aManifestMemberThatTheFormatDoesNotDefineIsRefused(testCase, undefinedMember)
            parts = FixedTools.editManifest(FixedTools.unpack(FixedTools.path('gru-pa')), undefinedMember{1}, undefinedMember{2});
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, parts)), 'opendpd:Package');
        end

        function aManifestValueOutsideWhatTheFormatAllowsIsRefusedAndSaysSo(testCase, badManifestValue)
            parts = FixedTools.editManifest(FixedTools.unpack(FixedTools.path('gru-pa')), badManifestValue{1}, badManifestValue{2});
            testCase.verifySubstring(refusalMessage(testCase, parts, 'opendpd:Package'), badManifestValue{3});
        end

        function theHashOfTheCheckpointIsOptionalAndMayBeNull(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            nulled = FixedTools.editManifest(parts, '"weights_sha256": "[0-9a-f]{64}"', '"weights_sha256": null');
            testCase.verifyEqual(opendpd.verify(opendpd.load(FixedTools.write(testCase, nulled))).status, 'bit_exact');
        end

        function theHashesThatIndexAGoldenVectorMustBeTheHashesOfItsFile(testCase, goldenHash)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            zeros64 = repmat('0', 1, 64);
            for id = ["normal", "extreme", "saturation", "all_zero", "state_reset", "long_sequence"]
                changed = FixedTools.editManifest(parts, ['("case_id": "' char(id) '"[^}]*?"' goldenHash '": ")[0-9a-f]{64}'], ['$1' zeros64]);
                testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, changed)), 'opendpd:PackageHash', char(id));
            end
        end

        function textInTheManifestIsMadePlainBeforeAnyoneSeesIt(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            changed = FixedTools.editManifest(parts, '"git_commit": null', '"git_commit": "a\\u001b[2Jb\\u0007c\\rd\\ne"');
            changed = FixedTools.editManifest(changed, '("case_id": "normal",\s*"description": ")[^"]*', '$1x\\u001b]0;title\\u0007y');
            model = opendpd.load(FixedTools.write(testCase, changed));
            testCase.verifyEqual(model.Manifest.software.git_commit, ['a?[2Jb?c?d' newline 'e']);
            testCase.verifyEqual(model.Manifest.golden(1).description, 'x?]0;title?y');
            testCase.verifyEqual(opendpd.verify(model).status, 'bit_exact');
        end

        function formatsBeyondWhatTheSixtyFourBitReferencesCanHoldAreRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            outputFraction = '("y": \{\s*"bits": 16,\s*"frac": )14';
            for fraction = [52 62]                         % the pre-activation holds 31 bits: 31 + (fraction - 20) reaches 63
                changed = FixedTools.editSpec(parts, outputFraction, ['$1 ' num2str(fraction)]);
                try
                    opendpd.load(FixedTools.write(testCase, changed));
                    message = '';
                catch cause
                    testCase.verifyEqual(cause.identifier, 'opendpd:Package', num2str(fraction));
                    message = cause.message;
                end
                testCase.verifySubstring(message, '64-bit', num2str(fraction));
            end
            % one place less is inside the references' range: the package loads (its golden vectors, made for 14, no longer fit)
            changed = FixedTools.editSpec(parts, outputFraction, '$1 51');
            report = opendpd.verify(opendpd.load(FixedTools.write(testCase, changed)));
            testCase.verifyFalse(report.passed);
            % a right shift of 63 places: weights with 62 fractional bits times inputs with 30, down to the pre-activation's 20
            wide = FixedTools.editJson(parts, "weights.json", '("fractions": \{\s*"w_ih": )15', '$1 62');
            wide = FixedTools.editManifest(wide, '("name": "w_ih",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )15', '$1 62');
            tooFar = FixedTools.editSpec(wide, '("x": \{\s*"bits": 16,\s*"frac": )14', '$1 30');
            try
                opendpd.load(FixedTools.write(testCase, tooFar));
                message = '';
            catch cause
                testCase.verifyEqual(cause.identifier, 'opendpd:Package');
                message = cause.message;
            end
            testCase.verifySubstring(message, '64-bit');
            exactly = FixedTools.editSpec(wide, '("x": \{\s*"bits": 16,\s*"frac": )14', '$1 20');      % 62 + 20 - 20: allowed
            testCase.verifyClass(opendpd.load(FixedTools.write(testCase, exactly)), 'opendpd.FixedModel');
            beyond = FixedTools.editSpec(wide, '("x": \{\s*"bits": 16,\s*"frac": )14', '$1 21');        % 62 + 21 - 20 = 63 places
            testCase.verifyError(@() opendpd.load(FixedTools.write(testCase, beyond)), 'opendpd:Package');
        end

        function anAccumulatorShiftBeyondSixtyTwoPlacesIsRefusedForEveryAccumulator(testCase, farShift)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            name = farShift;
            weights = FixedTools.editJson(parts, "weights.json", ['("fractions": \{[^}]*"' name '": )15'], '$1 62');
            weights = FixedTools.editManifest(weights, ['("name": "' name '",\s*"shape": \[[^\]]*\],\s*"bits": \d+,\s*"frac": )15'], '$1 62');
            for hFrac = [20 21]                       % 62 + 20 - 20 = 62 places are allowed, 62 + 21 - 20 = 63 are not
                changed = FixedTools.editSpec(weights, '("h": \{\s*"bits": 16,\s*"frac": )15', ['$1 ' num2str(hFrac)]);
                if hFrac == 21
                    testCase.verifySubstring(refusalMessage(testCase, changed, 'opendpd:Package'), '64-bit', name);
                else
                    try
                        opendpd.load(FixedTools.write(testCase, changed));
                        message = '';
                    catch cause
                        message = cause.message;
                    end
                    testCase.verifyFalse(contains(message, '64-bit'), message);        % refused later, if at all, for another reason
                end
            end
        end

        function tablesAndHiddenSizesBeyondTheLimitsAreRefused(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            wide = FixedTools.editSpec(parts, '"range": 8.0', '"range": 1000.0');                % 256000 entries on each side
            testCase.verifySubstring(refusalMessage(testCase, wide, 'opendpd:Package'), 'whole number of entries');
            for range = ["0.0", "-8.0", "8.001"]                                % no entries, no entries below zero, a part of an entry
                bad = FixedTools.editSpec(parts, '"range": 8.0', ['"range": ' char(range)]);
                testCase.verifySubstring(refusalMessage(testCase, bad, 'opendpd:Package'), 'whole number of entries', char(range));
            end
            longer = FixedTools.editSpec(parts, '"range": 8.0', '"range": 8.5');                  % a whole number of entries, not the file's
            testCase.verifySubstring(refusalMessage(testCase, longer, 'opendpd:Package'), 'has size');
            hidden = FixedTools.editJson(parts, "weights.json", '"hidden": 6', '"hidden": 513');
            testCase.verifySubstring(refusalMessage(testCase, hidden, 'opendpd:Package'), 'hidden must be a whole number from 1 to 512');
            fromTheManifest = FixedTools.editManifest(parts, '"hidden_size": 6', '"hidden_size": 513');
            testCase.verifySubstring(refusalMessage(testCase, fromTheManifest, 'opendpd:Package'), 'hidden_size must be a whole number from 1 to 512');
        end

        function theKernelEqualsTheInt64ReferenceAtTheEdgeOfTheReferencesRange(testCase)
            % The output fraction 31 places above the pre-activation's is the largest the references can hold (31 + 31 = 62
            % bits); the pre-activation at its extremes must saturate the output exactly as the int64 code does.
            f = FixedTools.randomFormat(RandStream('twister', Seed=21), true);
            f.preBits = 32; f.preMin = -2^31; f.preMax = 2^31 - 1; f.preFrac = 20;
            f.yBits = 16; f.yMin = -2^15; f.yMax = 2^15 - 1; f.yFrac = 51;
            f.bOut = [f.preMax; f.preMin];
            f.wOut(:) = 0;
            x = zeros(6, 2);
            [y, ~, trace] = opendpd.runtime.fixedGruRun(x, zeros(f.hidden, 1), zeros(1, 0), f);
            [yr, ~, tr] = FixedTools.referenceRun(f, x, zeros(f.hidden, 1), zeros(1, 0));
            testCase.verifyEqual(y, yr);
            testCase.verifyEqual(trace, tr);
            testCase.verifyEqual(y(1, :), [f.yMax f.yMin]);
        end

        function aResetListOfAnyLengthIsHandledInOnePass(testCase)
            n = 300;
            decided = false;
            for seed = 9:400                      % a format in which the state decides the first and the last output
                f = FixedTools.randomFormat(RandStream('twister', Seed=seed));
                stream = RandStream('twister', Seed=seed + 1000);
                x = round((rand(stream, n, 2) * 2 - 1) * f.xMax);
                h0 = round((rand(stream, f.hidden, 1) * 2 - 1) * f.hMax);
                plain = opendpd.runtime.fixedGruRun(x, h0, zeros(1, 0), f);
                zeroStart = opendpd.runtime.fixedGruRun(x(1, :), zeros(f.hidden, 1), zeros(1, 0), f);
                zeroEnd = opendpd.runtime.fixedGruRun(x(n, :), zeros(f.hidden, 1), zeros(1, 0), f);
                decided = ~isequal(plain(1, :), zeroStart) && ~isequal(plain(n, :), zeroEnd);
                if decided
                    break
                end
            end
            testCase.assertTrue(decided, 'no format with a state that matters at both ends was found');
            testCase.verifyEqual(opendpd.runtime.fixedGruRun(x, h0, 1, f), opendpd.runtime.fixedGruRun(x, zeros(f.hidden, 1), zeros(1, 0), f));
            resets = [7 7 1 n n + 5 1000 200 3 150 150 2 0 -3];             % repeated, unsorted, at the ends, beyond them and below
            [y, h, trace] = opendpd.runtime.fixedGruRun(x, h0, resets, f);
            [yr, hr, tr] = FixedTools.referenceRun(f, x, h0, resets);
            testCase.verifyEqual(y, yr);
            testCase.verifyEqual(h, hr);
            testCase.verifyEqual(trace, tr);
            long = [1:2:n, n + 1:1000];                                      % a reset before every other sample
            testCase.verifyEqual(opendpd.runtime.fixedGruRun(x, h0, long, f), FixedTools.referenceRun(f, x, h0, long));
        end

        function jsonShapeAgreesWithAScalarReferenceForAnyChunkSize(testCase)
            stream = RandStream('twister', Seed=11);
            bodies = {'a', '[', '{', ']', '}', ',', '\"', '\\', '\n', ' ', '\\\"'};
            outside = {'[', ']', '{', '}', ',', ':', ' ', '1', 'a', char(9), char(10), char(13)};   % no backslash outside a string
            for trial = 1:200
                pieces = cell(1, 0);
                for k = 1:randi(stream, 60)
                    if rand(stream) < 0.3
                        body = bodies(randi(stream, numel(bodies), 1, randi(stream, 8)));
                        pieces{end+1} = ['"' strjoin(body, '') '"']; %#ok<AGROW>
                    else
                        pieces{end+1} = outside{randi(stream, numel(outside))}; %#ok<AGROW>
                    end
                end
                text = strjoin(pieces, '');
                [depth, commas, token] = scalarShape(text);
                for chunk = [1 2 3 5 8 13 64 2^20]
                    [d, c, t] = opendpd.internal.jsonShape(uint8(text(:)), chunk);
                    testCase.verifyEqual([d c t], [depth commas token], sprintf('%s | chunk %d', text, chunk));
                end
            end
            % brackets and commas inside strings count for nothing; the last '[' opens a fourth level before an unclosed string
            [d, c, t] = opendpd.internal.jsonShape(uint8('{"a,b[": [1,2,{"c": "\\" ,["}]}').');
            testCase.verifyEqual([d c t], [4 3 1]);
            testCase.verifyEqual(opendpd.internal.jsonShape(uint8([]), 4), 0);
            [~, ~, token] = opendpd.internal.jsonShape(uint8('[1234567890, "12345678901234567890", -1.5e-7, true,false]').');
            testCase.verifyEqual(token, 10);                                  % the strings do not count; separators end a token
            [~, ~, token] = opendpd.internal.jsonShape(uint8(['[12' char(9) '345' char(10) '6789' char(13) '0]']).');
            testCase.verifyEqual(token, 4);                                   % tab, line feed and carriage return end a token
            for chunk = 1:12                                                   % a token cut by a chunk boundary is one token
                [~, ~, token] = opendpd.internal.jsonShape(uint8('[0.123456789,"x" , 99.5]').', chunk);
                testCase.verifyEqual(token, 11, sprintf('chunk %d', chunk));
            end
            % a string cut by a chunk boundary right after a backslash, and one right before the escaped quote
            text = uint8('["\\\\\"x]", [[1]]]').';
            for chunk = 1:numel(text)
                testCase.verifyEqual(opendpd.internal.jsonShape(text, chunk), 3, sprintf('chunk %d', chunk));
            end
        end

        function decodeJsonRefusesALongNumberBeforeJsondecodeSpendsTimeOnIt(testCase)
            % jsondecode needs time quadratic in the digits after the decimal point: a million digits take half a minute
            testCase.verifyEqual(opendpd.internal.decodeJson(uint8('[0.1234567890]').', 'num', MaxToken=12), 0.123456789);
            message = '';
            try
                opendpd.internal.decodeJson(uint8('[0.1234567890]').', 'num', MaxToken=11);
            catch cause
                message = cause.message;
                testCase.verifyEqual(cause.identifier, 'opendpd:Package');
            end
            testCase.verifySubstring(message, 'num holds a number or word of 12 characters');
            for text = {['[0.' repmat('1', 1, 100) ']'], ['{"a": 1e' repmat('9', 1, 100) '}'], ['[' repmat('t', 1, 100) ']'], ...
                    ['{"a": ' repmat('N', 1, 100) '}'], ['[' repmat('1', 1, 100) ']']}
                testCase.verifySubstring(refusalOf(@() opendpd.internal.decodeJson(uint8(text{1}).', 'long')), 'holds a number or word of');
            end
            % strings, keys and white space are not tokens, however long
            long = uint8(['{"' repmat('k', 1, 5000) '": "' repmat('v', 1, 5000) '",' repmat(' ', 1, 5000) '"b": 1}']).';
            testCase.verifyEqual(opendpd.internal.decodeJson(long, 'strings').b, 1);
        end

        function aLongNumberInAPackageIsRefusedFast(testCase)
            parts = FixedTools.unpack(FixedTools.path('gru-pa'));
            digits = repmat('1', 1, 9e5);                      % jsondecode alone takes about 30 seconds on this
            for file = ["weights.json", "manifest.json", "spec.json"]
                changed = FixedTools.editJson(parts, file, '"spec_id": "fixed-point-v1"', ['"spec_id": "fixed-point-v1", "junk": 0.' digits]);
                tic;
                message = refusalMessage(testCase, changed, 'opendpd:Package');
                testCase.verifyLessThan(toc, 10, char(file));
                testCase.verifySubstring(message, [char(file) ' holds a number or word of']);
            end
        end

        function aJsonErrorMessageIsShortEvenWhenJsondecodeQuotesTheWholeText(testCase)
            % jsondecode puts the bad token into its message: 100000 characters of it make a message of 100000 characters
            bytes = uint8(['[' repmat('t', 1, 1e5) ']']).';
            message = refusalOf(@() opendpd.internal.decodeJson(bytes, 'quoted', MaxToken=Inf));
            testCase.verifySubstring(message, 'quoted is not valid JSON');
            testCase.verifyLessThan(numel(message), 300);
        end

        function jsonIsReadAsUtf8(testCase)
            bytes = uint8([123 34 110 34 58 34 99 97 102 195 169 32 228 184 173 230 150 135 34 125]).';     % {"n":"cafe-acute Chinese"}
            value = opendpd.internal.decodeJson(bytes, 'utf8');
            testCase.verifyEqual(double(value.n), [99 97 102 233 32 20013 25991]);
        end

        function theLengthOfATokenIsCountedInCharactersNotBytes(testCase)
            % 40 accented letters outside a string are a 40-character token (80 bytes): not too long, but not JSON either
            bytes = uint8(['[' repmat([195 169], 1, 40) ']']).';
            testCase.verifySubstring(refusalOf(@() opendpd.internal.decodeJson(bytes, 'accents')), 'accents is not valid JSON');
            bytes = uint8(['[' repmat([195 169], 1, 65) ']']).';
            testCase.verifySubstring(refusalOf(@() opendpd.internal.decodeJson(bytes, 'accents')), 'number or word of 65 characters');
        end

        function jsonShapeIsTheSameOnTheBytesAndOnTheDecodedText(testCase)
            % decodeJson measures the text jsondecode will see; invalid UTF-8 must not move a bracket, a quote or a backslash
            stream = RandStream('twister', Seed=12);
            alphabet = uint8(['"\[]{},:01 a' 195 169 226 130 172 240 159 152 128 128 192 255 237 160]);
            for trial = 1:300
                bytes = alphabet(randi(stream, numel(alphabet), 1, randi(stream, 80))).';
                text = native2unicode(bytes.', 'UTF-8');
                [d1, c1, t1] = opendpd.internal.jsonShape(bytes);
                [d2, c2, t2] = opendpd.internal.jsonShape(uint8(text(:)));
                testCase.verifyEqual([d1 c1], [d2 c2], sprintf('trial %d', trial));
                testCase.verifyEqual(t1 > 0, t2 > 0);
            end
        end

        function plainTextReplacesControlCharactersAndNothingElse(testCase)
            first = struct('a', ['x' char(27) '[2Jy'], 'b', {{['t' char(8)], ['tab' char(9) 'ok' char(10) 'two']}}, ...
                'c', struct('d', ['e' char(27)]));
            second = struct('a', ['b' char([0 1 7 11 12 13 30 31 127 128 133 159]) 'c'], 'b', {{}}, 'c', 5);
            plain = opendpd.internal.plainText([first second]);
            testCase.verifyEqual(plain(1).a, 'x?[2Jy');
            testCase.verifyEqual(plain(1).b, {'t?', ['tab' char(9) 'ok' char(10) 'two']});     % tab and line feed stay
            testCase.verifyEqual(plain(1).c.d, 'e?');
            testCase.verifyEqual(plain(2).a, ['b' repmat('?', 1, 12) 'c']);        % 0, 1, 7, 11, 12, 13, 30, 31, 127, 128, 133, 159
            testCase.verifyEqual(plain(2).c, 5);
            international = [native2unicode(uint8([99 97 102 195 169 32 228 184 173 230 150 135]), 'UTF-8') char([160 161 255 256 8232])];
            testCase.verifyEqual(opendpd.internal.plainText(international), international);    % an accent, Chinese, NBSP and more
            testCase.verifyEqual(opendpd.internal.plainText({1, 'a', {}}), {1, 'a', {}});
        end

        % ----- the interface -----------------------------------------------------------------------------------------------
        function loadChoosesTheReaderFromTheEntryNames(testCase)
            testCase.verifyClass(opendpd.load(FixedTools.path('gru-pa')), 'opendpd.FixedModel');
            testCase.verifyClass(opendpd.load(PackageTools.path('gru-dpd')), 'opendpd.Model');
            testCase.verifyEqual(opendpd.internal.packageKind(FixedTools.path('gru-pa')), "fixed");
            testCase.verifyEqual(opendpd.internal.packageKind(PackageTools.path('gru-dpd')), "model");
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            junk = fullfile(folder, 'junk.zip');
            PackageTools.writeBytes(junk, uint8('junk'));
            testCase.verifyEqual(opendpd.internal.packageKind(junk), "model");
            % a model package that also carries the two JSON names is still a model package, and the model reader refuses it
            parts = PackageTools.unpack(PackageTools.path('gru-dpd'));
            parts = PackageTools.withEntry(PackageTools.withEntry(parts, "spec.json", uint8('{}')), "weights.json", uint8('{}'));
            mixed = PackageTools.write(testCase, parts);
            testCase.verifyEqual(opendpd.internal.packageKind(mixed), "model");
            testCase.verifyError(@() opendpd.load(mixed), 'opendpd:PackageContents');
        end

        function aFixedModelPrintsAndAnEmptyOneRefusesToRun(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            text = evalc('disp(model)');
            testCase.verifySubstring(text, 'opendpd.FixedModel');
            testCase.verifySubstring(text, 'fixed-point-v1');
            empty = opendpd.FixedModel();
            testCase.verifyError(@() empty.apply(ones(4, 1)), 'opendpd:Package');
            testCase.verifyError(@() empty.verify(), 'opendpd:Package');
        end

        function inputsThatAreNotIntegersOrIQAreRefused(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            testCase.verifyError(@() model.runInteger([1.5 2]), 'MATLAB:validators:mustBeInteger');
            testCase.verifyError(@() model.runInteger([1 2 3]), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.runInteger(zeros(0, 2)), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.runInteger([1 NaN]), 'MATLAB:validators:mustBeFinite');
            testCase.verifyError(@() model.runInteger([1 2], State=ones(5, 1)), 'opendpd:InvalidState');
            testCase.verifyError(@() model.runInteger([1 2], State=2^20 * ones(6, 1)), 'opendpd:InvalidState');
            testCase.verifyError(@() model.runInteger([1 2], ResetAt=0), 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() model.apply([1 NaN]), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.apply([]), 'opendpd:InvalidIQ');
            testCase.verifyError(@() model.apply(ones(3, 3)), 'opendpd:InvalidIQ');
        end

        function generateCodeDoesNotAcceptAFixedModel(testCase)
            model = opendpd.load(FixedTools.path('gru-pa'));
            folder = fullfile(testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder, 'out');
            testCase.verifyError(@() opendpd.generateCode(model, folder), 'opendpd:CodegenFixedPoint');
            testCase.verifyFalse(isfolder(folder));
        end

        function loadingDoesNotLoadPython(testCase)
            testCase.assumeFalse(strcmp(char(pyenv().Status), 'Loaded'), 'Python is already loaded in this session');
            opendpd.verify(opendpd.load(FixedTools.path('gru-pa')));
            testCase.verifyNotEqual(char(pyenv().Status), 'Loaded');
        end

        % ----- MATLAB Coder ---------------------------------------------------------------------------------------------------
        function theKernelBuildsWithMatlabCoderAndTheMexFunctionReproducesTheGoldenVectors(testCase, package)
            testCase.assumeTrue(license('test', 'MATLAB_Coder') && ~isempty(which('codegen')), 'MATLAB Coder is not available');
            testCase.assumeNotEmpty(mex.getCompilerConfigurations('C', 'Selected'), 'no C compiler is selected for MATLAB Coder');
            file = FixedTools.path(package);
            model = opendpd.load(file);
            read = opendpd.internal.readFixedPackage(file);
            f = model.kernelFormat();
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            testCase.applyFixture(matlab.unittest.fixtures.CurrentFolderFixture(folder));
            writelines(["function [y, h, trace] = fixedEntry(x, h, resetAt, f)", ...
                "[y, h, trace] = opendpd.runtime.fixedGruRun(x, h, resetAt, f);", "end"], 'fixedEntry.m');
            configuration = coder.config('mex');
            configuration.EnableJIT = false;
            codegen('fixedEntry', '-args', {coder.typeof(0, [Inf 2]), zeros(f.hidden, 1), coder.typeof(0, [1 Inf]), ...
                coder.Constant(f)}, '-config', configuration, '-o', 'fixedEntryMex', '-silent');
            testCase.addTeardown(@() clear('fixedEntryMex'));
            for entry = reshape(model.Manifest.golden, 1, [])
                g = read.golden.(entry.case_id);
                n = entry.n_samples;
                resets = reshape(entry.resets_at, 1, []) + 1;
                [y, ~, trace] = fixedEntryMex(double(reshape(g.x, 2, n).'), zeros(f.hidden, 1), resets, f);
                testCase.verifyEqual(y, double(reshape(g.y, 2, n).'), "outputs of " + entry.case_id);
                testCase.verifyEqual(trace, double(reshape(g.h_trace, f.hidden, n).'), "state steps of " + entry.case_id);
            end
        end
    end
end

function [maxDepth, commas, token] = scalarShape(text)
% The definition of jsonShape as a loop: nesting of brackets and braces, commas outside strings and the longest run of
% characters outside strings that are not white space, quotes or structure; a backslash in a string escapes the next character.
maxDepth = 0;
commas = 0;
token = 0;
run = 0;
depth = 0;
inString = false;
escaped = false;
for ch = text
    if inString
        if escaped
            escaped = false;
        elseif ch == '\'
            escaped = true;
        elseif ch == '"'
            inString = false;
        end
    elseif ch == '"'
        inString = true;
        run = 0;
    elseif any(ch == ' ' | ch == sprintf('\t') | ch == newline | ch == sprintf('\r') | ch == ':')
        run = 0;
    elseif ch == '[' || ch == '{'
        depth = depth + 1;
        maxDepth = max(maxDepth, depth);
        run = 0;
    elseif ch == ']' || ch == '}'
        depth = depth - 1;
        run = 0;
    elseif ch == ','
        commas = commas + 1;
        run = 0;
    else
        run = run + 1;
        token = max(token, run);
    end
end
end

function message = refusalMessage(testCase, parts, identifier)
% Load PARTS as a package, check that the error has IDENTIFIER, and return its message.
message = '';
try
    opendpd.load(FixedTools.write(testCase, parts));
    testCase.verifyFail('The package was not refused.');
catch cause
    testCase.verifyEqual(cause.identifier, identifier, cause.message);
    message = cause.message;
end
end

function parts = editPackage(parts, edit)
% EDIT is {where, pattern, replacement}: a text edit of the specification (and the manifest's copy), of weights.json or of the manifest.
switch edit{1}
    case 'spec'
        parts = FixedTools.editSpec(parts, edit{2}, edit{3});
    case 'weights'
        parts = FixedTools.editJson(parts, "weights.json", edit{2}, edit{3});
    otherwise
        parts = FixedTools.editManifest(parts, edit{2}, edit{3});
end
end

function message = refusalOf(call)
% The message of the opendpd:Package error that CALL raises (an empty message if it raises none).
message = '';
try
    call();
catch cause
    if strcmp(cause.identifier, 'opendpd:Package')
        message = cause.message;
    else
        message = ['unexpected ' cause.identifier ': ' cause.message];
    end
end
end
