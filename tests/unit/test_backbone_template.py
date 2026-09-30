"""Executable uploads, shape errors, resource bombs and fake approvals fail closed."""
import json

import pytest
import torch

from opendpd.core.backbone_template import (TEMPLATE, TemplateError, canonical_definition, parse_definition,
    scan_source, validate_definition)
from opendpd.core.registry import RegistryError, validate_parameters
from opendpd.core.template_network import TemplateNetwork
from scripts.check_user_backbone_review import approved_by_human, validate_changed_templates


@pytest.mark.parametrize("payload", [
    'import os\n', 'from os import system\n', 'print("executed")\n',
    'class Evil:\n    pass\n', 'def forward(x):\n    return x\n',
    'BACKBONE = __import__("os").environ\n', 'BACKBONE = dict()\n',
    'BACKBONE = {**{}}\n', 'BACKBONE = {"name": "a", "name": "b"}\n',
    'BACKBONE = [x for x in range(100000000)]\n', 'BACKBONE = "a" * 1000000000\n',
    'BACKBONE = ().__class__.__bases__\n', 'BACKBONE = {"nodes": lambda: None}\n',
    'BACKBONE = {"x": f"{1}"}\n', 'BACKBONE = {}\nBACKBONE = {}\n',
    'BACKBONE: dict = {}\n', '\ufeffBACKBONE = {}\n', '# \u202eevil\nBACKBONE = {}\n',
])
def test_executable_and_non_template_python_is_refused(payload):
    with pytest.raises(TemplateError):
        scan_source(payload.encode())


def test_no_upload_side_effects_and_bounded_file_names(tmp_path):
    marker = tmp_path / 'executed'
    with pytest.raises(TemplateError):
        scan_source(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'.encode())
    assert not marker.exists()
    for name in ('../safe.py', '/safe.py', 'safe.py.exe', 'safe.PY', 'a..py', '.py', 'a\\b.py'):
        with pytest.raises(TemplateError):
            scan_source(TEMPLATE.encode(), name)
    for source in (b'', b'#' * 32769, b'\xff', b'\x00', b'BACKBONE = ' + b'[' * 3000):
        with pytest.raises(TemplateError):
            scan_source(source)


def test_template_graph_is_canonical_and_registry_rejects_json_bypass():
    definition = scan_source(TEMPLATE.encode())
    encoded = canonical_definition(definition)
    assert parse_definition(encoded) == definition
    assert validate_parameters('user_template', {'definition': encoded}, 'pa') == {'definition': encoded}
    with pytest.raises(RegistryError):
        validate_parameters('user_template', {'definition': '{"op":"exec"}'}, 'dpd')
    with pytest.raises(TemplateError):
        parse_definition('[' * 10000)


@pytest.mark.parametrize('change', [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(output='input'),
    lambda d: d['nodes'][0].update(features=True),
    lambda d: d['nodes'][0].update(features=1000000),
    lambda d: d['nodes'][0].update(layers=3),
    lambda d: d['nodes'][0].update(inputs=['memory']),
    lambda d: d['nodes'][0].update(inputs=['project']),
    lambda d: d['nodes'][0].update(op='torch.load'),
    lambda d: d['nodes'][0].update(path='/etc/passwd'),
    lambda d: d['nodes'][1].update(features=3),
    lambda d: d['nodes'].append({'id':'unused','op':'identity','inputs':['input']}),
    lambda d: d['nodes'][1].update(id='memory'),
])
def test_shape_and_resource_validation(change):
    definition = scan_source(TEMPLATE.encode())
    change(definition)
    with pytest.raises(TemplateError):
        validate_definition(definition)


@pytest.mark.parametrize('op,options', [
    ('linear', {'features':8}), ('gru', {'features':8,'layers':2}), ('lstm', {'features':8}),
    ('conv1d', {'features':8,'kernel_size':5,'dilation':2}), ('layer_norm', {}),
    ('relu',{}), ('tanh',{}), ('gelu',{}), ('silu',{}), ('identity',{}), ('dropout',{'p':.2}), ('iq_features',{}),
])
def test_trusted_layers_have_correct_parameters_finite_gradients_and_no_lookahead(op, options):
    d = scan_source(TEMPLATE.encode())
    d['nodes'] = [{'id':'first','op':op,'inputs':['input'],**options},
                  {'id':'second','op':'linear','features':2,'inputs':['first']}]
    d['output'] = 'second'
    net = TemplateNetwork(canonical_definition(d)).eval()
    assert sum(p.numel() for p in net.parameters()) == validate_definition(d)['parameters']
    x = torch.randn(3,41,2,requires_grad=True)
    y = net(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    changed = x.detach().clone(); changed[:,20:] += 2
    torch.testing.assert_close(net(changed)[:,:20], y.detach()[:,:20], atol=1e-6, rtol=1e-5)
    zero = torch.zeros(1,3,2,requires_grad=True)
    net(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_residual_concat_and_no_unbounded_allocations():
    d = scan_source(TEMPLATE.encode())
    d['nodes'][2] = {'id':'combined','op':'concat','inputs':['input','project']}
    d['nodes'].append({'id':'result','op':'linear','features':2,'inputs':['combined']})
    assert TemplateNetwork(canonical_definition(d))(torch.randn(2,7,2)).shape == (2,7,2)
    d['nodes'] = [{'id':f'n{i}', 'op':'gru', 'features':128,'layers':2,'inputs':['input' if i==0 else f'n{i-1}']} for i in range(10)]
    d['nodes'].append({'id':'result','op':'linear','features':2,'inputs':['n9']})
    with pytest.raises(TemplateError, match='budget'):
        validate_definition(d)


def test_human_review_excludes_bots_author_comments_stale_dismissed_and_changes_requested():
    pr = {'head':{'sha':'a'*40}, 'user':{'login':'author'}}
    good = {'id':1,'commit_id':'a'*40,'state':'APPROVED','user':{'login':'reviewer','type':'User'}}
    assert approved_by_human(pr,[good],lambda _: 'maintain')
    assert not approved_by_human(pr,[good],lambda _: 'write')
    for update in ({'state':'COMMENTED'}, {'commit_id':'b'*40}, {'state':'DISMISSED'},
                   {'user':{'login':'author','type':'User'}}, {'user':{'login':'bot','type':'Bot'}}):
        assert not approved_by_human(pr,[{**good,**update}],lambda _: 'admin')
    assert not approved_by_human(pr,[good,{**good,'id':2,'state':'CHANGES_REQUESTED'}],lambda _: 'admin')
    assert not approved_by_human(pr,[good,{**good,'id':2,'state':'DISMISSED'}],lambda _: 'admin')
    assert approved_by_human(pr,[good,{**good,'id':2,'state':'COMMENTED'}],lambda _: 'admin')


def test_ci_scanner_checks_renames_and_hashes_without_importing_sources():
    from opendpd.services.user_backbones import package_manifest
    source = TEMPLATE.replace('"author": ""','"author": "Contributor"').encode()
    manifest = package_manifest(source)
    folder = f'backbones/user_uploaded/demo_{manifest["source_sha256"][:16]}'
    content = {f'{folder}/backbone.py':source, f'{folder}/manifest.json':json.dumps(manifest).encode()}
    files = [{'filename':name,'status':'added'} for name in content]
    validate_changed_templates(files,content.__getitem__)
    content[f'{folder}/backbone.py'] += b'\nimport os\n'
    with pytest.raises(TemplateError):
        validate_changed_templates(files,content.__getitem__)
    with pytest.raises(ValueError):
        validate_changed_templates([{'filename':'backbones/user_uploaded/__init__.py','status':'added'}],content.__getitem__)


def test_merge_queue_requires_exact_protected_bytes_and_fresh_human_approval():
    from scripts.check_user_backbone_review import evaluate_merge_group
    file={'filename':'opendpd/core/backbone_template.py','status':'modified','sha':'b'*40}
    pr={'number':4,'state':'open','changed_files':1,'base':{'ref':'main'},'head':{'sha':'c'*40},'user':{'login':'author'}}
    review={'id':1,'commit_id':'c'*40,'state':'APPROVED','user':{'login':'maintainer','type':'User'}}
    current_file=dict(file)
    def get(path):
        if path.startswith('compare/'): return {'files':[file],'total_commits':1}
        if path.startswith('pulls?'): return [{'number':4}]
        if path=='pulls/4': return pr
        if '/files?' in path: return [current_file]
        if '/reviews?' in path: return [review]
        if path.endswith('/permission'): return {'permission':'maintain'}
        raise AssertionError(path)
    group={'base_sha':'d'*40,'head_sha':'e'*40}
    assert evaluate_merge_group(group,get,lambda path:None)[0]
    current_file['sha']='f'*40
    assert not evaluate_merge_group(group,get,lambda path:None)[0]
    current_file['sha']=file['sha']; review['state']='DISMISSED'
    assert not evaluate_merge_group(group,get,lambda path:None)[0]
    review['state']='APPROVED'; review['commit_id']='0'*40
    assert not evaluate_merge_group(group,get,lambda path:None)[0]
