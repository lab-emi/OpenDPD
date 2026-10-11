function device = deviceFor(project, model, requested)
%DEVICEFOR Resolve Device="auto" with the service's own default; explicit devices pass through.
if requested ~= "auto"
    device = string(requested);
    return
end
module = bridge();
device = string(module.default_device(project.Backend, char(model)));
end
