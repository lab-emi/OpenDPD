"""The optional compiled forward retains gradients, AdamW and native inference."""
import copy
import pytest
import torch

from opendpd.core import arena, arena_engine as engine
from opendpd.core.arena_runner import configure_runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('key', ['pgjanet','dvrjanet','apnrru'])
def test_compiled_forward_preserves_zero_iq_tail_lr_and_adam(key):
    configure_runtime(key)
    torch.set_num_threads(1);torch.manual_seed(73)
    compiled=engine.build_model(key,arena.model_parameters(key,1000)).cuda()
    native=copy.deepcopy(compiled)
    native.forward=native._arena_eager_forward
    models=[native,compiled]
    opts=[torch.optim.AdamW(m.parameters(),lr=.005) for m in models]
    # All recurrence/phase/limiter math is exercised, with short records keeping
    # the regression test's compilation bounded. Production uses length 200.
    for index,batch in enumerate([4,3,4]):
        x=.1*torch.randn(batch,20,2,device='cuda');x[:,::5]=0
        if index==2:x.zero_()
        target=torch.randn_like(x)
        observations=[]
        for m,opt in zip(models,opts):
            m.train();opt.zero_grad();opt.param_groups[0]['lr']=.005*.5**index
            inputs=x.clone().requires_grad_();output=m(inputs)
            torch.nn.functional.mse_loss(output,target).backward()
            observations.append([output.detach(),inputs.grad,*[p.grad.clone() for p in m.parameters()]])
            opt.step()
        for a,b in zip(*observations):torch.testing.assert_close(a,b,rtol=2e-4,atol=8e-6)
        for a,b in zip(native.parameters(),compiled.parameters()):torch.testing.assert_close(a,b,rtol=2e-4,atol=8e-6)
        for a,b in zip(opts[0].state.values(),opts[1].state.values()):
            for field in a:torch.testing.assert_close(a[field],b[field],rtol=2e-4,atol=8e-6)
    native.load_state_dict(compiled.state_dict())
    native.eval();compiled.eval()
    with torch.no_grad():torch.testing.assert_close(native(x),compiled(x),rtol=0,atol=0)
