function kind = packageKind(file)
%PACKAGEKIND Which reader a package zip needs, from the names of its entries alone: "fixed" for a fixed-point-v1 deployment
% package (it has spec.json and weights.json), "model" for everything else, including files that are not readable zips, so
% that the model reader reports what is wrong with them. Nothing is extracted and no entry is read.
kind = "model";
try
    archive = java.util.zip.ZipFile(char(file));
catch
    return
end
closeArchive = onCleanup(@() archive.close());
names = strings(0, 1);
entries = archive.entries();
while entries.hasMoreElements()
    names(end+1, 1) = string(entries.nextElement().getName()); %#ok<AGROW>
end
if any(names == "spec.json") && any(names == "weights.json") && ~any(names == "weights.npz")
    kind = "fixed";
end
end
