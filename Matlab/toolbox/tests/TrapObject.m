classdef TrapObject
    % Test helper: loading this object from a MAT file runs loadobj, which leaves a marker file. TestModel uses it to show
    % that opendpd.load never loads objects from a package (the one way a MAT file can run code).
    properties
        Value = 1
    end
    methods (Static)
        function marker = markerFile()
            marker = fullfile(tempdir, 'opendpd-trap-fired.marker');
        end
        function obj = loadobj(obj)
            fid = fopen(TrapObject.markerFile(), 'w');
            fclose(fid);
        end
    end
end
