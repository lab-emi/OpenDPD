import torch
from torch import nn


class PGJANET(nn.Module):
    def __init__(self, hidden_size, output_size, bias=True):
        super(PGJANET, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias

        # Input gates
        self.W_a = nn.Linear(hidden_size + 1, hidden_size, bias=bias)  # a_n gate
        self.W_p1 = nn.Linear(hidden_size + 1, hidden_size, bias=bias)  # p1_n gate
        self.W_p2 = nn.Linear(hidden_size + 1, hidden_size, bias=bias)  # p2_n gate
        
        # Processing gates
        self.W_f = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)  # f_n gate
        self.W_g = nn.Linear(hidden_size + hidden_size, hidden_size, bias=bias)  # g_n gate
        
        # Output projection
        self.W_o = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x, h_0):
        # x shape: (batch_size, seq_len, 2)
        # h_0 shape: (num_layers, batch_size, hidden_size)
        
        batch_size, seq_len, _ = x.shape
        h = h_0[0]  # Take only the first layer's hidden state
        outputs = []

        # Process sequence step by step
        for t in range(seq_len):
            # Get current timestep input
            x_t = x[:, t, :]
            
            # Extract I/Q components
            i_x = x_t[:, 0].unsqueeze(-1)
            q_x = x_t[:, 1].unsqueeze(-1)
            amp_x = torch.sqrt(i_x**2 + q_x**2)
            
            # Calculate phase
            theta = torch.atan2(q_x, i_x)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # Input concatenation for gates
            h_x = torch.cat([h, amp_x], dim=-1)   
            h_cos = torch.cat([h, cos_theta], dim=-1)
            h_sin = torch.cat([h, sin_theta], dim=-1)

            # Calculate gates
            a_n = torch.tanh(self.W_a(h_x))
            p1_n = torch.tanh(self.W_p1(h_cos))
            p2_n = torch.tanh(self.W_p2(h_sin))

            # Calculate u_n (element-wise multiplication)
            u_n = a_n * p1_n * p2_n * (1 - a_n) * (1 - p1_n) * (1 - p2_n)

            # Process h_u concatenation
            h_u = torch.cat([h, u_n], dim=-1)

            # Calculate remaining gates
            f_n = torch.sigmoid(self.W_f(h_u))
            g_n = torch.tanh(self.W_g(h_u))

            # Calculate new hidden state
            h = f_n * h + (1 - f_n) * g_n

            # Calculate output
            y_n = self.W_o(h)
            outputs.append(y_n)

        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        
        return outputs

    def reset_parameters(self):
        for module in [self.W_a, self.W_p1, self.W_p2, self.W_f, self.W_g, self.W_o]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)