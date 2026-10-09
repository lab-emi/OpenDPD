classdef TestLab < matlab.unittest.TestCase
    % opendpd.lab.Session on the dry-run opendpd.lab.MockInstrument, with injected faults. Nothing here touches hardware and
    % nothing sets OPENDPD_ALLOW_RF_OUTPUT: the tests only ever read it. They do not need Python.
    properties
        U
    end
    properties (Constant)
        Fs = 80e6
        Gate = 'OPENDPD_ALLOW_RF_OUTPUT'
    end
    methods (TestClassSetup)
        function environment(testCase)
            PackageTools.assertToolboxUnderTest(testCase);
            testCase.assumeNotEqual(getenv(TestLab.Gate), '1', ...
                'The RF output gate is open in this shell (an approved lab session); these tests never run with it open.');
            testCase.U = 0.2 * PackageTools.signal(512, 5);
        end
    end
    methods (Test)
        function aNewSessionIsDisarmedAndWillNotMeasure(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester");
            testCase.verifyEqual(lab.State, "disarmed");
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:SafetyViolation');
            testCase.verifyEqual(inst.Measurements, 0, 'nothing reached the instrument');
            testCase.verifyFalse(inst.RFOn);
            testCase.verifyEqual(lab.State, "disarmed", 'a refusal is not a trip');
        end

        function armingNeedsAName(testCase)
            [lab, inst] = testCase.mock();
            testCase.verifyError(@() arm(lab), 'opendpd:lab:SafetyViolation');
            testCase.verifyError(@() arm(lab, "   "), 'opendpd:lab:SafetyViolation');
            try
                arm(lab);
            catch cause
                testCase.verifySubstring(cause.message, 'name');
            end
            testCase.verifyEqual(lab.State, "disarmed");
            arm(lab, "  Ann Example ");
            testCase.verifyEqual(lab.State, "armed");
            testCase.verifyEqual(lab.Operator, "Ann Example");
            armed = lab.Log(strcmp({lab.Log.Event}, 'armed'));
            testCase.verifyEqual(armed.Detail.operator, 'Ann Example');
            testCase.verifyError(@() arm(lab, "Someone Else"), 'opendpd:lab:SafetyViolation');
            testCase.verifyEqual(lab.Operator, "Ann Example", 'arming again does not change the operator');
            testCase.verifyFalse(inst.RFOn);
        end

        function aSessionBuiltFromFunctionsNeedsTheEnvironmentGate(testCase)
            calls = CallCounter();
            lab = opendpd.lab.Session(MeasureFcn=@calls.measure, RFOffFcn=@calls.rfOff, Operator="A. Tester");
            testCase.verifyFalse(lab.IsMock);
            testCase.verifyError(@() arm(lab), 'opendpd:lab:SafetyViolation');
            try
                arm(lab);
            catch cause
                testCase.verifySubstring(cause.message, 'OPENDPD_ALLOW_RF_OUTPUT');
            end
            testCase.verifyEqual(lab.State, "disarmed");
            testCase.verifyError(@() measure(lab, testCase.U), 'opendpd:lab:SafetyViolation');
            testCase.verifyEqual(calls.Count, 0, 'neither the measurement nor RF-off function was called');
        end

        function aSessionNeedsAnRFOffFunctionAndOneKindOfInstrument(testCase)
            calls = CallCounter();
            inst = opendpd.lab.MockInstrument();
            testCase.verifyError(@() opendpd.lab.Session(MeasureFcn=@calls.measure), 'opendpd:lab:Instrument');
            testCase.verifyError(@() opendpd.lab.Session(RFOffFcn=@calls.rfOff), 'opendpd:lab:Instrument');
            testCase.verifyError(@() opendpd.lab.Session(), 'opendpd:lab:Instrument');
            testCase.verifyError(@() opendpd.lab.Session(Instrument=struct()), 'opendpd:lab:Instrument');
            testCase.verifyError(@() opendpd.lab.Session(Instrument=inst, MeasureFcn=@calls.measure), 'opendpd:lab:Instrument');
            testCase.verifyError(@() opendpd.lab.Session(MeasureFcn=@calls.measure, RFOffFcn=@calls.rfOff, HeartbeatFcn=3), ...
                'opendpd:lab:Instrument');
            testCase.verifyTrue(meta.class.fromName('opendpd.lab.MockInstrument').Sealed, ...
                'a subclass that drives hardware must not be able to pass as the mock');
        end

        function limitsRefuseBeforeAnythingIsSent(testCase)
            peakPower = @(u) 10 * log10(max(abs(u)) ^ 2) + 30;                     % dBm for a 30 dB gain on peak power
            [lab, inst] = testCase.mock(Operator="A. Tester", MaxPeak=0.4, MaxOutputPower_dBm=20, PowerCalibration=peakPower);
            arm(lab);
            u = testCase.U / max(abs(testCase.U));
            bad = {0.5 * u, 'peak'; [0.1; NaN; 0.1], 'non-finite'; [0.1; Inf], 'non-finite'; complex(zeros(0, 1)), 'empty'; ...
                0.1 * ones(3, 3), 'vector'};
            for k = 1:size(bad, 1)
                testCase.verifyError(@() measure(lab, bad{k, 1}, SampleRate=testCase.Fs), 'opendpd:lab:SafetyViolation', ...
                    sprintf('case %d', k));
                testCase.verifyEqual(inst.Measurements, 0, sprintf('case %d: nothing sent', k));
                testCase.verifyEqual(lab.State, "armed", sprintf('case %d: a refusal is not a trip', k));
            end
            testCase.verifyError(@() measure(lab, 'abc', SampleRate=testCase.Fs), 'MATLAB:validators:mustBeNumeric');
            try
                measure(lab, 0.5 * u, SampleRate=testCase.Fs);
            catch cause
                testCase.verifySubstring(cause.message, 'Peak');
            end
            % the ceiling: peak 0.4 is allowed by MaxPeak but 22 dBm is above 20 dBm
            testCase.verifyError(@() measure(lab, 0.4 * u, SampleRate=testCase.Fs), 'opendpd:lab:SafetyViolation');
            try
                measure(lab, 0.4 * u, SampleRate=testCase.Fs);
            catch cause
                testCase.verifySubstring(cause.message, 'ceiling');
            end
            testCase.verifyEqual(inst.Measurements, 0);
            y = measure(lab, 0.3 * u, SampleRate=testCase.Fs);                       % 19.5 dBm: allowed
            testCase.verifySize(y, size(u));
            testCase.verifyEqual(inst.Measurements, 1);
            testCase.verifyEqual(lab.Measurements(1).power_dbm, peakPower(0.3 * u), AbsTol=1e-12);
        end

        function aCeilingNeverSilentlyDoesNotApply(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester", MaxOutputPower_dBm=10);
            arm(lab);
            u = testCase.U;
            testCase.verifyError(@() measure(lab, u, SampleRate=testCase.Fs), 'opendpd:lab:SafetyViolation');
            try
                measure(lab, u, SampleRate=testCase.Fs);
            catch cause
                testCase.verifySubstring(cause.message, 'must be known');
            end
            testCase.verifyError(@() measure(lab, u, SampleRate=testCase.Fs, RequestedPower_dBm=10.5), 'opendpd:lab:SafetyViolation');
            testCase.verifyEqual(inst.Measurements, 0);
            measure(lab, u, SampleRate=testCase.Fs, RequestedPower_dBm=9.5);
            testCase.verifyEqual(inst.Measurements, 1);
            % with a calibration and a stated power, the larger of the two is checked
            [lab, inst] = testCase.mock(Operator="A. Tester", MaxOutputPower_dBm=10, PowerCalibration=@(u) 5);
            arm(lab);
            testCase.verifyError(@() measure(lab, u, SampleRate=testCase.Fs, RequestedPower_dBm=11), 'opendpd:lab:SafetyViolation');
            measure(lab, u, SampleRate=testCase.Fs);
            testCase.verifyEqual(inst.Measurements, 1);
        end

        function aBrokenPowerCalibrationRefusesInsteadOfGuessing(testCase)
            bad = {@(u) error('cal:broken', 'no calibration file'), @(u) NaN, @(u) [1 2], @(u) 1 + 2i, @(u) Inf};
            for k = 1:numel(bad)
                [lab, inst] = testCase.mock(Operator="A. Tester", MaxOutputPower_dBm=30, PowerCalibration=bad{k});
                arm(lab);
                testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:SafetyViolation', ...
                    sprintf('case %d', k));
                testCase.verifyEqual(inst.Measurements, 0);
                testCase.verifyEqual(lab.State, "armed");
            end
        end

        function aMeasurementReturnsTheCaptureAndLeavesRFOff(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester");
            inst.NoiseRMS = 0;
            arm(lab);
            u = testCase.U;
            y = measure(lab, u, SampleRate=testCase.Fs);
            played = circshift(u, inst.DelaySamples);
            testCase.verifyEqual(y, inst.Gain * (played - inst.Cubic * abs(played) .^ 2 .* played), AbsTol=1e-12);
            testCase.verifyEqual(inst.LastInput, u, 'the instrument received the signal that was checked');
            testCase.verifyEqual(inst.LastSampleRate, testCase.Fs, 'and the sample rate');
            testCase.verifyClass(y, 'double');
            testCase.verifyFalse(isrow(y));
            testCase.verifyEqual(lab.State, "armed");
            testCase.verifyFalse(inst.RFOn, 'RF is off again as soon as the measurement is done');
            testCase.verifyGreaterThanOrEqual(inst.RFOffCalls, 1);
            % N-by-2 [I Q] in, N-by-2 [I Q] out, no sample rate stated
            inst.ReturnIQMatrix = true;
            z = measure(lab, [real(u) imag(u)]);
            testCase.verifyEqual(z, y + 0, AbsTol=1e-12);
            testCase.verifyEqual(inst.LastInput, u, 'an [I Q] matrix reaches the instrument as the complex signal');
            testCase.verifyTrue(isnan(inst.LastSampleRate), 'no sample rate stated, none passed');
            testCase.verifyTrue(isnan(lab.Measurements(2).sample_rate_hz));
            testCase.verifyEqual(lab.Measurements(1).sample_rate_hz, testCase.Fs);
            disarm(lab);
            testCase.verifyEqual(lab.State, "disarmed");
        end

        function rfCanBeHeldBetweenMeasurementsOnRequestAndGoesOffAtDisarm(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            arm(lab);
            measure(lab, testCase.U, SampleRate=testCase.Fs);
            testCase.verifyTrue(inst.RFOn, 'RFOffAfterMeasure=false keeps the output on for the next measurement');
            disarm(lab);
            testCase.verifyFalse(inst.RFOn);
            testCase.verifyEqual(lab.State, "disarmed");
            arm(lab);                                                                 % a disarmed session can be armed again
            testCase.verifyEqual(lab.State, "armed");
        end

        function anErrorInTheInstrumentTripsTheSession(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            arm(lab);
            inst.FailNext = "error";
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:MockOverload');
            testCase.verifyTripped(lab, inst, 'overload');
        end

        function aCaptureThatIsNotUsableTripsTheSession(testCase)
            for fault = ["nan", "empty", "matrix"]
                [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
                arm(lab);
                inst.FailNext = fault;
                testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:BadCapture', char(fault));
                testCase.verifyTripped(lab, inst, 'unusable');
            end
        end

        function aMeasurementThatTakesTooLongTripsTheSession(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester", Timeout=0.2, RFOffAfterMeasure=false);
            inst.SlowSeconds = 0.6;
            inst.FailNext = "slow";
            arm(lab);
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:Timeout');
            testCase.verifyTripped(lab, inst, 'limit is 0.2 s');
        end

        function aLostLinkTripsTheSession(testCase)
            % the link is gone before the session is armed: arming fails and nothing else happens
            [lab, inst] = testCase.mock(Operator="A. Tester");
            inst.LoseLinkAfter = 0;
            testCase.verifyError(@() arm(lab), 'opendpd:lab:LinkLost');
            testCase.verifyEqual(lab.State, "disarmed");
            % lost before the measurement: the instrument is never asked to play
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            inst.LoseLinkAfter = 1;
            arm(lab);
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:LinkLost');
            testCase.verifyEqual(inst.Measurements, 0);
            testCase.verifyTripped(lab, inst, 'lost');
            % lost right after the measurement: the capture is not returned
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            inst.LoseLinkAfter = 2;
            arm(lab);
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:LinkLost');
            testCase.verifyEqual(inst.Measurements, 1);
            testCase.verifyEmpty(lab.Measurements, 'a capture taken over a lost link is not recorded');
            testCase.verifyTripped(lab, inst, 'lost');
        end

        function aTrippedSessionStaysTripped(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester");
            arm(lab);
            inst.FailNext = "error";
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:MockOverload');
            testCase.verifyError(@() arm(lab), 'opendpd:lab:SafetyViolation');
            testCase.verifyError(@() arm(lab, "Another Operator"), 'opendpd:lab:SafetyViolation');
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:SafetyViolation');
            try
                arm(lab);
            catch cause
                testCase.verifySubstring(cause.message, 'overload');
            end
            disarm(lab);
            testCase.verifyEqual(lab.State, "tripped", 'disarm switches RF off but does not clear a trip');
            abort(lab, "again");
            testCase.verifyEqual(lab.TripReason, "mock analyser overload", 'the first reason is kept');
            testCase.verifyEqual(inst.Measurements, 1);
        end

        function ifRFOffFailsTheSessionSaysSoLoudly(testCase)
            % while tripping on another failure: the original error is what you get, plus a warning
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            arm(lab);
            inst.FailNext = "error";
            inst.RFOffFails = true;
            testCase.verifyWarning(@() testCase.swallow(@() measure(lab, testCase.U, SampleRate=testCase.Fs)), 'opendpd:lab:RFOffFailed');
            testCase.verifyEqual(lab.State, "tripped");
            testCase.verifyTrue(lab.RFOffFailed);
            testCase.verifyTrue(inst.RFOn, 'the mock really is still on: the session must have told us');
            testCase.verifyTrue(lab.record().rf_off_failed);
            % after a successful measurement: not a silent success
            [lab, inst] = testCase.mock(Operator="A. Tester");
            arm(lab);
            inst.RFOffFails = true;
            testCase.verifyWarning(@() testCase.swallow(@() measure(lab, testCase.U, SampleRate=testCase.Fs)), 'opendpd:lab:RFOffFailed');
            testCase.verifyError(@() testCase.quietly(@() measure(lab, testCase.U, SampleRate=testCase.Fs)), 'opendpd:lab:SafetyViolation');
            testCase.verifyEqual(lab.State, "tripped");
            % disarm that cannot switch RF off
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            arm(lab);
            inst.RFOffFails = true;
            testCase.verifyWarning(@() testCase.swallow(@() disarm(lab)), 'opendpd:lab:RFOffFailed');
            testCase.verifyEqual(lab.State, "tripped");
            testCase.verifyError(@() testCase.quietly(@() disarm(lab)), 'opendpd:lab:RFOffFailed');
        end

        function abortTripsAndDeletingAnArmedSessionSwitchesRFOff(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            arm(lab);
            measure(lab, testCase.U, SampleRate=testCase.Fs);
            testCase.verifyTrue(inst.RFOn);
            abort(lab, "operator pressed stop");
            testCase.verifyEqual(lab.State, "tripped");
            testCase.verifyEqual(lab.TripReason, "operator pressed stop");
            testCase.verifyFalse(inst.RFOn);
            [lab, inst] = testCase.mock(Operator="A. Tester", RFOffAfterMeasure=false);
            arm(lab);
            measure(lab, testCase.U, SampleRate=testCase.Fs);
            testCase.verifyTrue(inst.RFOn);
            calls = inst.RFOffCalls;
            delete(lab);
            testCase.verifyFalse(inst.RFOn, 'deleting an armed session switches RF off');
            testCase.verifyGreaterThan(inst.RFOffCalls, calls);
            % a disarmed session that is deleted does not touch the instrument again
            [lab, inst] = testCase.mock(Operator="A. Tester");
            calls = inst.RFOffCalls;
            delete(lab);
            testCase.verifyEqual(inst.RFOffCalls, calls);
        end

        function theRecordIsCompleteAndHashesMatchPython(testCase)
            % hash of the float32 I/Q bytes, as opendpd.core.measurement.to_iq(z).tobytes() hashes them in Python
            z = complex((1:8).' / 16, (-4:3).' / 16);
            testCase.verifyEqual(opendpd.lab.iqHash(z), '45b6a1346b0daf8b0e93e0faf712d6245e4325c632d4007cbf186d66041d8f86');
            [lab, inst] = testCase.mock(Operator="A. Tester", MaxPeak=0.8, MaxOutputPower_dBm=25, PowerCalibration=@(u) 12.5, Timeout=5);
            arm(lab);
            u = testCase.U;
            y1 = measure(lab, u, SampleRate=testCase.Fs);
            y2 = measure(lab, 0.5 * u, SampleRate=testCase.Fs);
            disarm(lab);
            r = lab.record();
            testCase.verifyEqual(r.format, 'opendpd-lab-session-v1');
            testCase.verifyTrue(r.mock);
            testCase.verifyEqual(r.operator, 'A. Tester');
            testCase.verifyEqual(r.final_state, 'disarmed');
            testCase.verifyEqual(r.limits.max_peak_abs, 0.8);
            testCase.verifyEqual(r.limits.max_output_power_dbm, 25);
            testCase.verifyEqual(r.limits.timeout_s, 5);
            testCase.verifyEqual(numel(r.measurements), 2);
            testCase.verifyEqual(r.measurements(1).played_sha256, opendpd.lab.iqHash(u));
            testCase.verifyEqual(r.measurements(1).captured_sha256, opendpd.lab.iqHash(y1));
            testCase.verifyEqual(r.measurements(2).played_sha256, opendpd.lab.iqHash(0.5 * u));
            testCase.verifyEqual(r.measurements(2).captured_sha256, opendpd.lab.iqHash(y2));
            testCase.verifyEqual(r.measurements(1).n_played, numel(u));
            testCase.verifyEqual(r.measurements(1).power_dbm, 12.5);
            testCase.verifyEqual(r.measurements(1).peak, max(abs(u)), AbsTol=1e-12);
            events = string({r.log.Event});
            testCase.verifyEqual(events(1:3), ["created", "armed", "measure"]);
            testCase.verifyEqual(events(end), "disarmed");
            again = jsondecode(jsonencode(r));
            testCase.verifyEqual(again.measurements(2).played_sha256, r.measurements(2).played_sha256);
            testCase.verifyGreaterThanOrEqual(inst.RFOffCalls, 3);
        end

        function aTripIsWrittenToTheRecord(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester");
            arm(lab);
            inst.FailNext = "error";
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:MockOverload');
            r = lab.record();
            testCase.verifyEqual(r.final_state, 'tripped');
            testCase.verifySubstring(r.trip_reason, 'overload');
            events = string({r.log.Event});
            testCase.verifyTrue(any(events == "tripped"));
            testCase.verifyTrue(any(events == "rf_off"));
            testCase.verifyEmpty(r.measurements);
        end

        function theCompactRecordFitsTheDatasetNotesAndPointsAtTheFullOne(testCase)
            [lab, inst] = testCase.mock(Operator="Ann Example With A Rather Long Name", RFOffAfterMeasure=false);
            arm(lab);
            for k = 1:8
                measure(lab, k / 10 * testCase.U, SampleRate=testCase.Fs, RequestedPower_dBm=3.5);
            end
            inst.FailNext = "error";
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:MockOverload');
            lab2 = testCase.longReason();
            for session = [lab, lab2]
                compact = session.record(Compact=true);
                testCase.verifyLessThan(strlength(jsonencode(compact)), 1500, 'importIQ refuses a Source above 1500 characters');
                testCase.verifyTrue(compact.compact);
                testCase.verifyEqual(compact.final_state, 'tripped');
                file = fullfile(string(testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder), "sessions", "one.json");
                info = session.saveRecord(file);
                testCase.verifyEqual(compact.record_sha256, info.sha256, 'the compact form names the saved file by its SHA-256');
                testCase.verifyEqual(opendpd.internal.sha256(info.file), info.sha256);
                saved = jsondecode(fileread(info.file));
                testCase.verifyEqual(saved.format, 'opendpd-lab-session-v1');
                testCase.verifyEqual(saved.final_state, 'tripped');
            end
            testCase.verifyEqual(compact.n_measurements, 0);
            testCase.verifyEqual(strlength(compact.trip_reason), 203, 'a long reason is cut to 200 characters and marked');
            testCase.verifyTrue(endsWith(string(compact.trip_reason), "..."));
            testCase.verifyEqual(strlength(lab2.record().trip_reason), 900, 'the complete record keeps all of it');
            first = lab.record(Compact=true);
            testCase.verifyEqual(first.n_measurements, 8);
            testCase.verifyEqual(first.last.played_sha256, opendpd.lab.iqHash(0.8 * testCase.U));
            testCase.verifyEqual(first.last.sample_rate_hz, testCase.Fs);
            testCase.verifyLessThanOrEqual(strlength(first.trip_reason), 203);
        end

        function theCompactRecordIsBoundedWhateverTheFreeTextIs(testCase)
            % The operator name and the trip reason are free text. The compact record is what travels in a dataset's notes
            % (2000 characters, shared with the importer's own entries), so neither may make it grow without limit.
            [lab, inst] = testCase.mock(Operator=string(repmat('N', 1, 400)));
            arm(lab);
            for k = 1:3
                measure(lab, testCase.U, SampleRate=testCase.Fs, RequestedPower_dBm=1);
            end
            lab.abort(string(repmat('r', 1, 5000)));
            compact = lab.record(Compact=true);
            testCase.verifyEqual(strlength(compact.operator), 103);
            testCase.verifyTrue(endsWith(string(compact.operator), "..."));
            testCase.verifyEqual(strlength(compact.trip_reason), 203);
            testCase.verifyLessThan(strlength(jsonencode(compact)), 1500);
            testCase.verifyEqual(strlength(lab.record().operator), 400, 'the complete record keeps the whole name');
            testCase.verifyEqual(strlength(lab.record().trip_reason), 5000);
            testCase.verifyFalse(inst.RFOn);
        end

        function theDisplayShowsTheStateAtAGlance(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester");
            text = evalc('disp(lab)');
            testCase.verifySubstring(text, 'DISARMED');
            testCase.verifySubstring(text, 'mock');
            arm(lab);
            inst.FailNext = "error";
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:MockOverload');
            text = evalc('disp(lab)');
            testCase.verifySubstring(text, 'TRIPPED');
            testCase.verifySubstring(text, 'overload');
        end

        function theLabCodeNeverSetsTheGateOrUsesPython(testCase)
            folder = fullfile(fileparts(fileparts(which('opendpd.lab.Session'))), '+lab');
            files = dir(fullfile(folder, '*.m'));
            testCase.assertNotEmpty(files);
            for k = 1:numel(files)
                text = fileread(fullfile(files(k).folder, files(k).name));
                testCase.verifyEmpty(regexp(text, 'setenv\s*\(', 'once'), [files(k).name ' sets an environment variable']);
                testCase.verifyEmpty(regexp(text, '(?<![A-Za-z0-9_.])py\.', 'once'), [files(k).name ' uses Python']);
            end
            tests = dir(fullfile(fileparts(mfilename('fullpath')), '*.m'));
            for k = 1:numel(tests)
                text = fileread(fullfile(tests(k).folder, tests(k).name));
                testCase.verifyEmpty(regexp(text, 'setenv\s*\(\s*[''"]OPENDPD_ALLOW_RF_OUTPUT', 'once'), ...
                    [tests(k).name ' sets the RF output gate']);
            end
        end
    end

    methods (Access = private)
        function [lab, inst] = mock(~, options)
            arguments
                ~
                options.Operator (1,1) string = ""
                options.MaxPeak (1,1) double = 1
                options.MaxOutputPower_dBm (1,1) double = NaN
                options.PowerCalibration = []
                options.Timeout (1,1) double = 30
                options.RFOffAfterMeasure (1,1) logical = true
            end
            inst = opendpd.lab.MockInstrument();
            args = namedargs2cell(options);
            lab = opendpd.lab.Session(args{:}, Instrument=inst);
        end

        function lab = longReason(testCase)
            [lab, inst] = testCase.mock(Operator="A. Tester");
            arm(lab);
            lab.abort(string(repmat('x', 1, 900)));
            testCase.verifyEqual(lab.State, "tripped");
            testCase.verifyFalse(inst.RFOn);
        end

        function verifyTripped(testCase, lab, inst, reasonPart)
            testCase.verifyEqual(lab.State, "tripped");
            testCase.verifySubstring(char(lab.TripReason), reasonPart);
            testCase.verifyFalse(inst.RFOn, 'RF is off');
            testCase.verifyGreaterThanOrEqual(inst.RFOffCalls, 1);
            testCase.verifyError(@() arm(lab), 'opendpd:lab:SafetyViolation');
            testCase.verifyError(@() measure(lab, testCase.U, SampleRate=testCase.Fs), 'opendpd:lab:SafetyViolation');
            disarm(lab);
            testCase.verifyEqual(lab.State, "tripped");
            r = lab.record();
            testCase.verifyEqual(r.final_state, 'tripped');
            testCase.verifySubstring(r.trip_reason, reasonPart);
        end
    end

    methods (Static, Access = private)
        function swallow(fcn)
            try
                fcn();
            catch
            end
        end

        function quietly(fcn)
            state = warning('off', 'opendpd:lab:RFOffFailed');
            restore = onCleanup(@() warning(state));
            fcn();
        end
    end
end
