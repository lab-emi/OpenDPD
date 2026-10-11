function id = datasetID(dataset)
if isstruct(dataset) && isscalar(dataset) && isfield(dataset, 'dataset_id')
    id = string(dataset.dataset_id);
else
    id = string(dataset);
end
validateattributes(id, {'string'}, {'scalar', 'nonempty'});
end
