classdef CallCounter < handle
    % Counts calls. The lab tests use it to prove that a refused measurement never reached the instrument functions.
    properties
        Count (1,1) double = 0
    end
    methods
        function y = measure(obj, u, varargin)
            obj.Count = obj.Count + 1;
            y = u;
        end
        function rfOff(obj)
            obj.Count = obj.Count + 1;
        end
    end
end
