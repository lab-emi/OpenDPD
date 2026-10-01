"""Full-window coverage, restart equivalence and validation-only selection."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from opendpd.core import arena, arena_engine as engine


def test_training_window_view_keeps_last_frame_and_never_wraps():
    x = np.arange(207*2,dtype=np.float32).reshape(207,2)
    frames = engine.training_frames(x,'cpu')
    assert tuple(frames.shape) == (8,200,2)
    for start in range(8):
        np.testing.assert_array_equal(frames[start].numpy(), x[start:start+200])
    assert arena.training_budget(264)['optimizer_updates'] == 480  # 65 windows: 64 + 1
    with pytest.raises(ValueError, match='shorter'):
        arena.training_budget(199)


def test_interrupted_full_training_resumes_exact_optimizer_schedule_and_shuffle(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(arena,'TRAINING',{**arena.TRAINING,'epochs':5,'batch_size':3})

    class Gain(nn.Module):
        def __init__(self,value):
            super().__init__();self.weight=nn.Parameter(torch.tensor(value))
        def forward(self,x):
            return self.weight*x

    class TrainValOnly(dict):
        def __getitem__(self,key):
            assert key in ('x_train','x_val')
            return super().__getitem__(key)

    x=np.random.default_rng(23).normal(0,.1,(207,2)).astype(np.float32)
    def condition():
        return SimpleNamespace(identifier='restart-check',device='cpu',teacher=Gain(2.),
            gain=1.2,peak=10.,data=TrainValOnly(x_train=x,x_val=x[:205]))
    monkeypatch.setattr(engine,'build_model',lambda *_:Gain(.15))
    def objective(c,y):
        values=engine.validation_metrics(y,c.gain*c.data['x_val'])
        return dict(values,objective_db=values['nmse_db'])
    monkeypatch.setattr(engine,'validation_objective',objective)
    whole=tmp_path/'whole';whole.mkdir()
    restarted=tmp_path/'restarted';restarted.mkdir()
    model,info=engine.train_gradient(condition(),'gru',{},7,whole,lambda *_:None)
    def interrupt(phase,epoch,message):
        if epoch==3: raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError,match='simulated interruption'):
        engine.train_gradient(condition(),'gru',{},7,restarted,interrupt)
    assert (restarted/'resume.pt').exists() and not (restarted/'training.json').exists()
    resumed,record=engine.train_gradient(condition(),'gru',{},7,restarted,lambda *_:None)
    assert torch.equal(model.weight,resumed.weight)
    for key in ('optimizer_updates','frame_draw_sha256','selected_epoch','selected_validation_objective_db'):
        assert info[key]==record[key]
    assert info['optimizer_updates']==15 and info['frame_exposures']==40
    assert not (restarted/'resume.pt').exists()


def test_spectral_validation_uses_only_original_validation_context(monkeypatch):
    from opendpd.core.metrics import spectral_v2
    from opendpd.schemas import SignalSpec
    seen=[]
    x=np.random.default_rng(4).normal(0,.1,(6327,2)).astype(np.float32)
    c=SimpleNamespace(gain=1.,data={'x_val':x},manifest={'signal':SignalSpec(
        sample_rate_hz=983040000.,bandwidth_hz=200000000.,n_sub_ch=5,nperseg=19662).model_dump()})
    def spectral(y,reference,signal):
        seen.append((y.copy(),reference.copy(),signal.nperseg))
        return [SimpleNamespace(name=k,value=v) for k,v in [('IBE',-40.),('ACLR_L',-50.),('ACLR_R',-60.)]]
    monkeypatch.setattr(spectral_v2,'compute',spectral)
    result=engine.validation_objective(c,x)
    assert result['objective_db']==-45. and seen[0][2]==4096
    np.testing.assert_array_equal(seen[0][0],x[200:-200])
    np.testing.assert_array_equal(seen[0][1],x[200:-200])
