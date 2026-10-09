classdef TestMetrics < matlab.unittest.TestCase
    % Real MATLAB/Python integration for opendpd.waveform and opendpd.metrics.*. Set OPENDPD_MATLAB_PYTHON before running.
    % The expected numbers are the ones registered in docs/performance/matlab-parity.md (signal S4, nperseg 2048): the
    % metrics here are the code Studio uses, so they must reproduce what the cross-validation recorded.
    properties
        Waveform
        Capture
    end
    properties (Constant)
        WaveformSHA256 = 'a972712353f4c2b6068c4ad96f8bf7fdb0f2df5c14ca136b203a587047ec2102'
        RecordedEVM = 5.315247162741446
        RecordedACLR_L = -32.837638463302845
        RecordedACLR_R = -32.61101131042427
    end
    methods (TestClassSetup)
        function environment(testCase)
            executable = string(getenv('OPENDPD_MATLAB_PYTHON'));
            testCase.assertNotEmpty(char(executable), ...
                'Set OPENDPD_MATLAB_PYTHON to the Python environment with this checkout installed.');
            info = opendpd.setup(PythonExecutable=executable);
            testCase.assertTrue(info.ok);
            testCase.Waveform = opendpd.waveform(Seed=1, Subframes=10);
            testCase.Capture = TestMetrics.signalS4(testCase.Waveform.x);
        end
    end
    methods (Test)
        function waveformIsReproducible(testCase)
            w = testCase.Waveform;
            testCase.verifySize(w.x, [307200 1]);
            testCase.verifyEqual(w.SampleRate, 30.72e6);
            testCase.verifyEqual(mean(abs(w.x).^2), 1, RelTol=1e-3);
            testCase.verifyEqual(w.SHA256, testCase.WaveformSHA256);
            testCase.verifySize(w.Symbols, [140 1200]);
            again = opendpd.waveform(Seed=1, Subframes=10);
            testCase.verifyEqual(again.x, w.x);
            other = opendpd.waveform(Seed=2, Subframes=1);
            testCase.verifySize(other.x, [30720 1]);
            testCase.verifyNotEqual(other.SHA256, w.SHA256);
        end

        function waveformArgumentsAreValidated(testCase)
            testCase.verifyError(@() opendpd.waveform(Subframes=0), 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() opendpd.waveform(Seed=-1), 'MATLAB:validators:mustBeNonnegative');
        end

        function symbolsReproduceTheWaveformWithLTEToolbox(testCase)
            testCase.assumeTrue(license('test', 'LTE_Toolbox') && ~isempty(which('lteOFDMModulate')), 'LTE Toolbox not available');
            w = testCase.Waveform;
            enb = struct('NDLRB', 100, 'CyclicPrefix', 'Normal', 'Windowing', 0);
            x = lteOFDMModulate(enb, w.Symbols.');
            testCase.verifySize(x, size(w.x));
            scale = real(sum(conj(x) .* w.x)) / real(sum(abs(x).^2));
            % Budget registered in docs/performance/matlab-parity.md (P1): 1e-6 after one real scale factor.
            testCase.verifyLessThan(norm(w.x - scale * x) / norm(w.x), 1e-6);
        end

        function evmOfTheCleanWaveform(testCase)
            w = testCase.Waveform;
            m = opendpd.metrics.evm(w.x, w, SampleRate=w.SampleRate);
            testCase.verifyEqual(m.Status, 'ok');
            testCase.verifyLessThan(m.EVM_RMS, 1e-4);            % float32 storage of the waveform is the floor
            testCase.verifyEqual(m.EVM_dB, 20 * log10(m.EVM_RMS / 100), RelTol=1e-9);
        end

        function metricsReproduceTheRegisteredParityValues(testCase)
            y = testCase.Capture;
            m = opendpd.metrics.evm(y, testCase.Waveform, SampleRate=122.88e6);
            testCase.verifyEqual(m.Status, 'ok');
            testCase.verifyEqual(m.EVM_RMS, testCase.RecordedEVM, RelTol=1e-6);
            a = opendpd.metrics.aclr(y, SampleRate=122.88e6, SegmentSamples=2048, Waveform=testCase.Waveform);
            testCase.verifyEqual(a.Status, 'ok');
            testCase.verifyEqual(a.ACLR_L, testCase.RecordedACLR_L, AbsTol=1e-4);
            testCase.verifyEqual(a.ACLR_R, testCase.RecordedACLR_R, AbsTol=1e-4);
        end

        function aclrMovesWithTheSegmentLength(testCase)
            y = testCase.Capture;
            short = opendpd.metrics.aclr(y, SampleRate=122.88e6, SegmentSamples=1024, Waveform=testCase.Waveform);
            long = opendpd.metrics.aclr(y, SampleRate=122.88e6, SegmentSamples=4096, Waveform=testCase.Waveform);
            % The reason SegmentSamples has no default: the ratio depends on the resolution.
            testCase.verifyNotEqual(short.ACLR_L, long.ACLR_L);
            testCase.verifyLessThan(abs(short.ACLR_L - long.ACLR_L), 1);
        end

        function requiredMetadataHasNoDefault(testCase)
            y = testCase.Capture;
            testCase.verifyError(@() opendpd.metrics.aclr(y, SampleRate=122.88e6), 'opendpd:SignalMetadata');
            testCase.verifyError(@() opendpd.metrics.aclr(y, SegmentSamples=2048), 'opendpd:SignalMetadata');
            testCase.verifyError(@() opendpd.metrics.evm(y, testCase.Waveform), 'opendpd:SignalMetadata');
            try
                opendpd.metrics.aclr(y, SampleRate=122.88e6);
            catch cause
                testCase.verifySubstring(cause.message, 'Welch segment length');
            end
        end

        function theWrongWaveformIsRefusedNotScored(testCase)
            other = opendpd.waveform(Seed=2, Subframes=10);
            m = opendpd.metrics.evm(testCase.Capture, other, SampleRate=122.88e6);
            testCase.verifyEqual(m.Status, 'missing_reference');
            testCase.verifyTrue(isnan(m.EVM_RMS));
            testCase.verifyNotEmpty(m.Reason);
        end

        function evaluateReturnsATable(testCase)
            t = opendpd.metrics.evaluate(testCase.Capture, SampleRate=122.88e6, SegmentSamples=2048, Waveform=testCase.Waveform);
            testCase.verifyClass(t, 'table');
            testCase.verifyEqual(t.Properties.VariableNames, {'Name', 'Value', 'Unit', 'Status', 'Reason'});
            testCase.verifyEqual(t.Name, ["EVM_RMS"; "EVM_DB"; "ACLR_L"; "ACLR_R"]);
            testCase.verifyEqual(t.Status, repmat("ok", 4, 1));
            testCase.verifyEqual(t.Unit, ["%"; "dB"; "dBc"; "dBc"]);
            % The profile is defined for a capture of its waveform: without it nothing is scored, nothing is guessed.
            bare = opendpd.metrics.evaluate(testCase.Capture, SampleRate=122.88e6, SegmentSamples=2048);
            testCase.verifyEqual(bare.Status, repmat("missing_reference", 4, 1));
            testCase.verifyTrue(all(isnan(bare.Value)));
            testCase.verifySubstring(char(bare.Reason(1)), 'reference waveform');
        end

        function aWaveformStructMustComeFromOpendpdWaveform(testCase)
            testCase.verifyError(@() opendpd.metrics.evaluate(testCase.Capture, SampleRate=122.88e6, ...
                SegmentSamples=2048, Waveform=struct('foo', 1)), 'opendpd:Waveform');
        end

        function carrierAclrProfile(testCase)
            a = opendpd.metrics.aclr(testCase.Capture, SampleRate=122.88e6, SegmentSamples=2048, ...
                Profile="opendpd-spectral-v2", Bandwidth=18e6);
            testCase.verifyEqual(a.Status, 'ok');
            testCase.verifyLessThan(a.ACLR_L, 0);
            testCase.verifyLessThan(a.ACLR_R, 0);
        end
    end
    methods (Static)
        function y = signalS4(x30)
            % The registration's S4: the waveform interpolated x4 (FFT zero padding), drive 0.35, the synthetic PA of
            % tests/fixtures/synthetic.py (memory polynomial, lags 0..2), stored as single like an OpenDPD dataset.
            n = numel(x30);
            a = fft(x30);
            up = ifft([a(1:n/2); zeros(3 * n, 1); a(n/2+1:n)]) * 4;
            u = 0.35 * up;
            terms = {1, 0, 1.0; 1, 1, 0.05 - 0.02i; 1, 2, -0.01i; 3, 0, -0.35 + 0.08i; 3, 1, -0.04; 5, 0, 0.05 - 0.01i};
            y = zeros(size(u));
            for k = 1:size(terms, 1)
                lag = terms{k, 2};
                shifted = [zeros(lag, 1); u(1:end-lag)];
                y = y + terms{k, 3} * shifted .* abs(shifted) .^ (terms{k, 1} - 1);
            end
            y = single(y);
        end
    end
end
