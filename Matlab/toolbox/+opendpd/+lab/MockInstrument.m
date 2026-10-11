classdef (Sealed) MockInstrument < handle
    % A dry-run generator/analyser pair around a fixed synthetic PA: no hardware, no RF (the counterpart of OpenDPD's
    % Python mock adapter). Use it to learn the session procedure and to test your scripts; what it returns is never
    % evidence about a power amplifier.
    %   inst = opendpd.lab.MockInstrument();
    %   lab = opendpd.lab.Session(Instrument=inst, Operator="Your Name");
    % Like a real chain it keeps playing after a measurement until RF is turned off: RFOn stays true until rfOff is
    % called, so a session's behaviour can be checked. The class is sealed: a session treats exactly this class as one
    % that emits nothing, so a subclass that talks to hardware cannot be passed off as a mock.
    % Faults can be injected, to test that a script leaves RF off when something goes wrong:
    %   FailNext      "error" (the analyser overloads), "nan" (a capture with a non-finite sample), "empty" (nothing
    %                 captured), "matrix" (a capture of the wrong shape), "slow" (the call takes SlowSeconds), or ""
    %                 for none; it applies to the next measurement only
    %   ReturnIQMatrix  return the capture as an N-by-2 [I Q] matrix instead of a complex vector
    %   LoseLinkAfter heartbeats that answer before the link is lost (Inf: never)
    %   RFOffFails    rfOff throws (a broken link at the worst moment)
    properties
        Gain (1,1) double = 2 * exp(1i * deg2rad(30))
        Cubic (1,1) double = 0.12
        NoiseRMS (1,1) double = 1e-3
        DelaySamples (1,1) double {mustBeInteger, mustBeNonnegative} = 37
        Seed (1,1) double = 0
        FailNext (1,1) string {mustBeMember(FailNext, ["", "error", "nan", "empty", "matrix", "slow"])} = ""
        ReturnIQMatrix (1,1) logical = false
        SlowSeconds (1,1) double = 2
        LoseLinkAfter (1,1) double = Inf
        RFOffFails (1,1) logical = false
    end
    properties (SetAccess = private)
        RFOn (1,1) logical = false
        RFOffCalls (1,1) double = 0
        Measurements (1,1) double = 0
        Beats (1,1) double = 0
        LastInput = []                 % the signal and sample rate the last measurement received (NaN: none stated)
        LastSampleRate (1,1) double = NaN
    end
    properties (Constant)
        IsMock = true
    end
    methods
        function y = measure(obj, u, sampleRate)
            % Play u (looped) and capture as many samples: the output stays on afterwards, like a real generator.
            obj.RFOn = true;
            obj.Measurements = obj.Measurements + 1;
            obj.LastInput = u;
            obj.LastSampleRate = NaN;
            if nargin > 2
                obj.LastSampleRate = sampleRate;
            end
            fault = obj.FailNext;
            obj.FailNext = "";
            if fault == "error"
                error('opendpd:lab:MockOverload', 'mock analyser overload');
            elseif fault == "slow"
                pause(obj.SlowSeconds);
            end
            v = u(:);
            played = circshift(v, obj.DelaySamples);
            y = obj.Gain * (played - obj.Cubic * abs(played) .^ 2 .* played);
            noise = RandStream('twister', Seed=obj.Seed + obj.Measurements);
            y = y + obj.NoiseRMS / sqrt(2) * complex(randn(noise, numel(y), 1), randn(noise, numel(y), 1));
            switch fault
                case "nan"
                    y(max(1, floor(end / 2))) = NaN;
                case "empty"
                    y = complex(zeros(0, 1));
                case "matrix"
                    y = repmat(y, 1, 3);
                otherwise
                    if obj.ReturnIQMatrix
                        y = [real(y) imag(y)];
                    end
            end
        end

        function heartbeat(obj)
            obj.Beats = obj.Beats + 1;
            if obj.Beats > obj.LoseLinkAfter
                error('opendpd:lab:LinkLost', 'mock instrument stopped answering');
            end
        end

        function rfOff(obj)
            obj.RFOffCalls = obj.RFOffCalls + 1;
            if obj.RFOffFails
                error('opendpd:lab:MockRFOff', 'mock instrument could not switch RF off');
            end
            obj.RFOn = false;
        end

        function info = describe(~)
            info = struct('name', "mock", 'description', ...
                "dry-run generator/analyser around a fixed cubic stand-in PA; emits nothing", 'mock', true);
        end
    end
end
