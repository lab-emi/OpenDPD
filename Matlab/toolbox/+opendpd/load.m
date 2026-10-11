function model = load(file)
%LOAD Load a model package as an object that plain MATLAB code can run.
% model = opendpd.load("apa-dpd.opendpd.zip") reads an opendpd-model-v1 package as data (no Python, no project, no code from the
% archive is executed), checks every file against the manifest's SHA-256, and returns an opendpd.Model. A fixed-point-v1
% deployment package (written by `opendpd deploy` or the Studio's Deployment panel) is read the same way and returns an
% opendpd.FixedModel. Use opendpd.verify(model) to check it against the package's golden test vectors on this MATLAB release,
% and y = opendpd.apply(model, x) to run it. Create a model package from a trained run with opendpd.export(job, file).
arguments
    file (1,1) string
end
if opendpd.internal.packageKind(file) == "fixed"
    model = opendpd.FixedModel(opendpd.internal.readFixedPackage(file), file);
    return
end
package = opendpd.internal.readPackage(file);
model = opendpd.Model(package, file);
end
