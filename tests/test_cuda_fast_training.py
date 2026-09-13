"""CUDA replay must preserve updates, losses, LR changes and frozen PA weights."""
import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import CoreModel, CascadedModel
from modules.train_funcs import net_train


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("key,pa", [("gru", None), ("dgru", None), ("lstm", None),
                                      ("tres_gru", None), ("rvtdcnn", None),
                                      ("gru", "gru"), ("gru", "dgru"), ("tres_gru", "dgru")])
def test_replay_preserves_training_with_partial_batches_and_lr_changes(key, pa, monkeypatch):
    torch.manual_seed(221)
    original = CoreModel(2, 8, 1, key).cuda()
    if pa:
        original = CascadedModel(original, CoreModel(2, 8, 1, pa).cuda())
        original.freeze_pa_model()
    accelerated = copy.deepcopy(original)
    for net in (original, accelerated):
        for module in net.modules():
            if isinstance(module, nn.RNNBase):
                module.flatten_parameters()
    frozen = {n: p.clone() for n, p in original.named_parameters() if not p.requires_grad}
    data = TensorDataset(torch.randn(39, 40, 2), torch.randn(39, 40, 2))
    loader = DataLoader(data, batch_size=16, shuffle=False, pin_memory=True)
    opts = [torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.005)
            for net in (original, accelerated)]
    for epoch in range(3):
        losses = []
        for index, (net, opt) in enumerate(zip((original, accelerated), opts)):
            opt.param_groups[0]['lr'] = .005 * (.5 ** epoch)
            monkeypatch.setenv('OPENDPD_DISABLE_CUDA_FAST_PATH', '1' if index == 0 else '0')
            log = {}
            rng = torch.cuda.get_rng_state().clone()
            net_train(log, net, loader, opt, nn.MSELoss(), .2, torch.device('cuda'))
            assert torch.equal(rng, torch.cuda.get_rng_state())
            losses.append(log['loss'])
        assert losses[0] == pytest.approx(losses[1], rel=2e-6, abs=1e-7)
        for a, b in zip(original.parameters(), accelerated.parameters()):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        for opt_a, opt_b in zip(opts[0].state.values(), opts[1].state.values()):
            for k in opt_a:
                torch.testing.assert_close(opt_a[k], opt_b[k], rtol=2e-5, atol=2e-6)
    fast = opts[1]._opendpd_fast_step
    assert fast.graph is not None and not fast.failed
    for name, before in frozen.items():
        assert torch.equal(dict(accelerated.named_parameters())[name], before)


def test_cpu_and_unreviewed_models_keep_eager_path():
    from modules.cuda_fast_training import make_fast_step
    model = nn.Linear(2, 2)
    assert make_fast_step(model, nn.MSELoss(), tuple(model.parameters()), 0) is None
