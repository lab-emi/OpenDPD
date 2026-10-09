function result = generateCode(model, folder, options)
%GENERATECODE Write a loaded model package as a standalone MATLAB class for MATLAB Coder and Simulink.
%   result = opendpd.generateCode(model, folder) writes four text files into FOLDER (created if needed):
%     <Name>.m        a matlab.System with the weights as constants and the model's arithmetic in the code-generation
%                     subset, so that it works as a Simulink MATLAB System block and with MATLAB Coder;
%     <Name>Step.m    the MATLAB Coder entry point;
%     <Name>Check.m   runs the class on the package's golden test vector and reports the error;
%     README_<Name>.md  the provenance (package SHA-256, run) and the limits.
%   The class needs no OpenDPD toolbox, no Python and no data file: the numerical kernels are the text of opendpd.runtime.
%   result = opendpd.generateCode(model, folder, Name="MyDPD", Execution="streaming_stateful", Overwrite=true)
%     Name       class name; default OpenDPD<Role><Model>, for example OpenDPDDpdGru. A name that is already a function,
%                class or file on the path is refused, so that a generated file cannot shadow another.
%     Execution  "offline_segmented" (default, how the run was scored: the state resets every nperseg samples) or
%                "streaming_stateful" (gru and gmp: the state is carried from call to call, reset(obj) clears it).
%     Overwrite  replace files that opendpd.generateCode wrote earlier; any other file is never replaced.
%   The result has Name, Folder, Files, Execution, Model and PackageSHA256. The same package, name and execution always give
%   the same bytes. The arithmetic is double precision (single-rounded inputs), not fixed-point and not HDL-ready.
arguments
    model (1,1) opendpd.Model
    folder (1,1) string {mustBeNonzeroLengthText}
    options.Name (1,1) string = ""
    options.Execution (1,1) string {mustBeMember(options.Execution, ...
        ["offline_segmented", "streaming_stateful", "streaming"])} = "offline_segmented"
    options.Overwrite (1,1) logical = false
end
data = model.codegenInputs();
execution = options.Execution;
if execution == "streaming"
    execution = "streaming_stateful";
end
name = options.Name;
if name == ""
    name = defaultName(data.Manifest);
end
target = absoluteFolder(folder);
checkName(name, target);
sources = opendpd.internal.generateSources(data, char(name), execution);     % raises before anything is written
files = [target + filesep + name + ".m"
    target + filesep + name + "Step.m"
    target + filesep + name + "Check.m"
    target + filesep + "README_" + name + ".md"];
texts = {sources.class, sources.step, sources.check, sources.readme};
refuseToReplace(files, options.Overwrite);
if ~isfolder(target)
    mkdir(target);
end
written = strings(0, 1);
try
    for i = 1:numel(files)
        writeText(files(i), texts{i});
        written(end+1, 1) = files(i); %#ok<AGROW>
    end
catch cause
    for file = written'
        delete(file);
    end
    rethrow(cause);
end
result = struct('Name', char(name), 'Folder', char(target), 'Files', files, 'Execution', char(execution), ...
    'Model', data.Manifest.model.key, 'PackageSHA256', data.SHA256);
end

function name = defaultName(manifest)
role = regexprep(char(manifest.run.role), '[^A-Za-z0-9]', '');
key = char(manifest.model.key);
parts = strsplit(key, '_');
camel = '';
for i = 1:numel(parts)
    camel = [camel upper(parts{i}(1)) parts{i}(2:end)]; %#ok<AGROW>
end
if isempty(role)
    role = 'model';
end
name = string(['OpenDPD' upper(role(1)) role(2:end) camel]);
end

function folder = absoluteFolder(folder)
% The folder as an absolute path without creating it (a relative path is taken from the current folder).
folder = char(folder);
if ~(startsWith(folder, filesep) || ~isempty(regexp(folder, '^[A-Za-z]:[\\/]', 'once')) || startsWith(folder, '\\'))
    folder = fullfile(pwd, folder);
end
folder = string(regexprep(folder, '[\\/]+$', ''));
end

function checkName(name, target)
if ~isvarname(name) || strlength(name) > 40
    error('opendpd:CodegenName', ['"%s" cannot be a class name here: use a MATLAB identifier of at most 40 ' ...
        'characters.'], name);
end
for candidate = [name, name + "Step", name + "Check"]
    found = which(char(candidate));
    if ~isempty(found) && ~startsWith(string(found), target + filesep, IgnoreCase=ispc)
        error('opendpd:CodegenName', ['"%s" is already a function, class or file on the MATLAB path (%s); a generated ' ...
            'file must not shadow it. Choose another Name.'], candidate, found);
    end
end
end

function refuseToReplace(files, overwrite)
mark = opendpd.internal.codegenMark();
for file = files'
    if isfile(file)
        if ~overwrite
            error('opendpd:CodegenExists', '%s exists; use Overwrite=true to replace a file this function wrote.', file);
        end
        if ~contains(fileread(file), mark)
            error('opendpd:CodegenForeignFile', '%s was not written by opendpd.generateCode and is not replaced.', file);
        end
    elseif isfolder(file)
        error('opendpd:CodegenExists', '%s is a folder.', file);
    end
end
end

function writeText(file, text)
fid = fopen(file, 'w', 'n', 'UTF-8');
if fid < 0
    error('opendpd:CodegenWrite', 'Cannot write %s.', file);
end
closer = onCleanup(@() fclose(fid));
count = fwrite(fid, text, 'char');
if count ~= numel(text)
    error('opendpd:CodegenWrite', 'Could not write all of %s.', file);
end
clear closer
end
