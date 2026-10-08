function model = load(file)
%LOAD Load an opendpd-model-v1 package as an opendpd.Model that plain MATLAB code can run.
% model = opendpd.load("apa-dpd.opendpd.zip") reads the package as data (no Python, no project, no code from the
% archive is executed), checks every file against the manifest's SHA-256, and returns the model. Use
% opendpd.verify(model) to check it against the package's golden test vector on this MATLAB release, and
% y = opendpd.apply(model, x) to run it. Create a package from a trained run with opendpd.export(job, file).
arguments
    file (1,1) string
end
package = opendpd.internal.readPackage(file);
model = opendpd.Model(package, file);
end
