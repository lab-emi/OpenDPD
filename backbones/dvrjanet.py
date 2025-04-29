import torch
from torch import nn


class DVRJANET(nn.Module):
    def __init__(self, hidden_size, output_size, num_dvr_units=4, bias=True):
        super(DVRJANET, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_dvr_units = num_dvr_units
        self.bias = bias

        # Phase Recurrent Filter
        self.W_ph = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_pθ = nn.Linear(1, hidden_size, bias=False)

        # Magnitude Recurrent Filter
        self.W_ah = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ax = nn.Linear(1, hidden_size, bias=False)
        # DVR parameters
        self.cs = nn.Parameter(torch.randn(num_dvr_units))

        # Processing gates
        self.W_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_ccos = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)
        self.W_csin = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)

        # Output projections
        self.W_o1 = nn.Linear(hidden_size, 1, bias=bias)
        self.W_o2 = nn.Linear(hidden_size, 1, bias=bias)

    def dvr_block(self, x, cs):
        # Implement DVR block as shown in figure (b)
        outputs = []
        k=1
        num_k = len(cs)
        for c in cs:
            out = torch.abs(x - k/num_k) * c
            outputs.append(out)
            k+=1
        return sum(outputs)

    def forward(self, x, h_0):
        # x shape: (batch_size, seq_len, 2)
        # h_0 shape: (2, batch_size, hidden_size) - for I/Q components
        
        batch_size, seq_len, _ = x.shape
        h_I = h_0.squeeze(0) # I component hidden state
        h_Q = h_0.squeeze(0) # Q component hidden state
        outputs_I = []
        outputs_Q = []

        for t in range(seq_len):
            # Get current timestep input
            x_t = x[:, t, :]
            i_x = x_t[:, 0].unsqueeze(-1)
            q_x = x_t[:, 1].unsqueeze(-1)
            
            # Calculate magnitude and phase
            magnitude = torch.sqrt(i_x**2 + q_x**2)
            theta = torch.atan2(q_x, i_x)

            # Phase Recurrent Filter
            θ_tilde = self.W_pθ(theta) + self.W_ph(h_I + h_Q)
            
            # Magnitude Recurrent Filter with DVR
            a_tilde = self.dvr_block(
                self.W_ax(magnitude) + self.W_ah(h_I + h_Q),
                self.cs
            )

            # Calculate trigonometric components
            cos_θ = torch.cos(θ_tilde)
            sin_θ = torch.sin(θ_tilde)

            # Calculate gates as per equation (10)
            h_combined = h_I + h_Q
            f_n = torch.sigmoid(self.W_f(h_combined))

            # Calculate g components
            h_a_cos = torch.cat([h_I, a_tilde * cos_θ], dim=-1)
            h_a_sin = torch.cat([h_Q, a_tilde * sin_θ], dim=-1)
            g_cos_n = torch.tanh(self.W_ccos(h_a_cos))
            g_sin_n = torch.tanh(self.W_csin(h_a_sin))

            # Update hidden states
            h_I = f_n * h_I + (1 - f_n) * g_cos_n
            h_Q = f_n * h_Q + (1 - f_n) * g_sin_n

            # Calculate outputs
            y_I = self.W_o1(h_I)
            y_Q = self.W_o2(h_Q)
            
            outputs_I.append(y_I)
            outputs_Q.append(y_Q)

        # Stack outputs along sequence dimension
        outputs_I = torch.stack(outputs_I, dim=1)
        outputs_Q = torch.stack(outputs_Q, dim=1)
        outputs = torch.cat([outputs_I, outputs_Q], dim=-1)
        outputs = outputs.view(batch_size, seq_len, self.output_size)
        return outputs

    def reset_parameters(self):
        for module in [self.W_ph, self.W_pθ, self.W_ah, self.W_ax, 
                      self.W_f, self.W_ccos, self.W_csin, 
                      self.W_o1, self.W_o2]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        