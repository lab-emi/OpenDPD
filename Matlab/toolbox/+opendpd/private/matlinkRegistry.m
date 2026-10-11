function output = matlinkRegistry(action, key, link)
%MATLINKREGISTRY Retain live bridge handles even when studio has no output.
persistent links
if isempty(links), links = containers.Map('KeyType', 'char', 'ValueType', 'any'); end
output = [];
allKeys = keys(links);
for index = 1:numel(allKeys)
    item = links(allKeys{index});
    if ~isvalid(item), remove(links, allKeys{index}); end
end
switch action
    case 'get'
        if isKey(links, key), output = links(key); end
    case 'put'
        links(key) = link;
    case 'all'
        output = values(links);
    case 'remove'
        if isKey(links, key) && (nargin < 3 || links(key) == link), remove(links, key); end
end
end
