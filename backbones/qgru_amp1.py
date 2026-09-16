"""Amplitude-feature variant of the shared quantized GRU backbone."""
import torch
from backbones.qgru import QGRU as BaseQGRU


class QGRU(BaseQGRU):
    def forward(self, x, h_0):
        # Feature Extraction
        i_x = torch.unsqueeze(x[..., 0], dim=-1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        # amp2 = self.pow2(i_x) + self.pow2(q_x)
        amp = torch.sqrt(amp2)
        # amp = self.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        # amp3 = self.pow3(amp)
        
        x = torch.cat((i_x, q_x, amp, amp3), dim=-1)


        # Regressor
        out, _ = self.rnn(x, h_0)
        out = self.fc_out(out)
        return out
