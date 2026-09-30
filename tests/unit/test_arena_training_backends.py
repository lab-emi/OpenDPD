"""Dense training identities, native inference, and isolated seed execution."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from opendpd.core import arena, arena_engine as engine, arena_runner as runner

DEVICES = [('cpu', torch.float64), ('cpu', torch.float32), pytest.param(
    'cuda', torch.float32, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))]


@pytest.mark.parametrize('key', ['deltagru', 'deltajanet', 'tres_deltagru', 'bojanet'])
@pytest.mark.parametrize('device,dtype', DEVICES)
@pytest.mark.parametrize('zero', [False, True])
def test_dense_delta_preserves_native_gradients_adam_and_export(key, device, dtype, zero):
    torch.manual_seed(9)
    dense = engine.build_model(key, arena.model_parameters(key,1000)).to(device=device,dtype=dtype)
    native = copy.deepcopy(dense)
    layer=native.backbone if key=='bojanet' else native.backbone.rnn
    layer.forward = layer._arena_eager_forward
    if key=='tres_deltagru':layer.use_triton=False
    models = [native, dense]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.005) for model in models]
    tolerance = dict(rtol=1e-8, atol=1e-10) if dtype==torch.float64 else dict(rtol=1e-4, atol=7e-6)
    # FP32 reductions can perturb gradients close to Adam's epsilon (1e-8).
    # Keep forward/gradient bounds tight; bound the resulting update separately.
    update_tolerance = tolerance if dtype==torch.float64 else dict(rtol=1e-3,atol=1e-4)
    for index, batch in enumerate([3, 2, 3]):
        x=.1*torch.randn(batch,200,2,device=device,dtype=dtype)
        if zero:x.zero_()
        target=torch.randn_like(x)
        results=[]
        for model,opt in zip(models,optimizers):
            model.train();opt.zero_grad();opt.param_groups[0]['lr']=.005*.5**index
            input=x.clone().requires_grad_()
            state=x.new_zeros((1,batch,model.hidden_size)) if key=='bojanet' else None
            output=model(input,state)
            nn.functional.mse_loss(output,target).backward()
            results.append([output.detach(),input.grad,*[p.grad.clone() for p in model.parameters()]])
            opt.step()
        # Subsequent observations include the preceding FP32 Adam differences.
        observed_tolerance=tolerance if dtype==torch.float64 or index==0 else dict(rtol=2e-4,atol=3e-5)
        for a,b in zip(*results):torch.testing.assert_close(a,b,**observed_tolerance)
        for a,b in zip(native.parameters(),dense.parameters()):torch.testing.assert_close(a,b,**update_tolerance)
        for a,b in zip(optimizers[0].state.values(),optimizers[1].state.values()):
            for field in a:torch.testing.assert_close(a[field],b[field],**tolerance)
    assert list(native.state_dict())==list(dense.state_dict())
    native.load_state_dict(dense.state_dict())
    with torch.no_grad():
        # Both .eval() and no_grad() in training mode retain native execution.
        for training in [False,True]:
            native.train(training);dense.train(training)
            if key=='tres_deltagru':dense.backbone.rnn.use_triton=False
            torch.testing.assert_close(native(x,state),dense(x,state),rtol=0,atol=0)


def test_nonzero_delta_thresholds_retain_native_training():
    from opendpd.core.arena_training_backends import dense_training_module
    for key in ['deltagru','deltajanet']:
        model=engine.build_model(key,dict(hidden_size=8,num_layers=1,thx=0,thh=0))
        model.backbone.rnn.th_x=model.backbone.rnn.th_h=.1
        assert not dense_training_module(model.backbone.rnn)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_parallel_seed_prefetch_matches_serial_with_partial_batch(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENDPD_DISABLE_ARENA_COMPILE','1')
    monkeypatch.setattr(arena,'TRAINING',{**arena.TRAINING,'epochs':3,'threads':1})
    torch.manual_seed(49)
    teacher=engine.build_model('tres_gru',dict(hidden_size=4,num_layers=1)).cuda()
    x=np.random.default_rng(13).normal(0,.1,(264,2)).astype(np.float32)
    contexts=[]
    class TrainValOnly(dict):
        def __getitem__(self,key):
            assert key in ('x_train','x_val')
            return super().__getitem__(key)
    def condition(identifier,device):
        c=SimpleNamespace(identifier=identifier,device=device,teacher=copy.deepcopy(teacher),
            gain=1.2,peak=1.,data=TrainValOnly(x_train=x,x_val=x))
        contexts.append(c)
        return c
    monkeypatch.setattr(runner,'TrainingCondition',condition)
    def objective(c,y):
        values=engine.validation_metrics(y,c.gain*c.data['x_val'])
        return dict(values,objective_db=values['nmse_db'])
    monkeypatch.setattr(engine,'validation_objective',objective)
    for mode in ['0','1']:
        monkeypatch.setenv('OPENDPD_ARENA_PARALLEL_SEEDS',mode)
        runner.prefetch('tres_gru',250,'private-seeds',tmp_path/mode,device='cuda')
    assert len(contexts)==4 and len({id(c.teacher) for c in contexts})==4
    params=arena.model_parameters('tres_gru',250)
    for seed in arena.SEEDS:
        folders=[runner.cache_folder(tmp_path/mode,'tres_gru',params,'private-seeds',seed) for mode in ['0','1']]
        with np.load(folders[0]/'weights.npz') as a,np.load(folders[1]/'weights.npz') as b:
            for name in a.files:np.testing.assert_allclose(a[name],b[name],rtol=3e-5,atol=3e-6)
        records=[runner.read_json(folder/'training.json') for folder in folders]
        for field in ['optimizer_updates','frame_exposures','frame_draw_sha256','selected_epoch']:
            assert records[0][field]==records[1][field]
        assert records[0]['optimizer_updates']==6
