import torch
from torch import nn


class BOJANET(nn.Module):
    def __init__(self, hidden_size, output_size, bias=True):
        super(BOJANET, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.window_size = 16  # M in the figure
        self.num_vd_units = 6  # P in the figure
        self.bias = bias

        # FIR Filters Layer parameters
        self.fir_I = nn.Linear(self.window_size, self.num_vd_units, bias=False)  # b_{p,0} and b_{p,1}
        self.fir_Q = nn.Linear(self.window_size, self.num_vd_units, bias=False)  # b_{p,0} and b_{p,1}
        # Recursive Topology Layer
        self.W_fi = nn.Linear(self.num_vd_units*2, hidden_size, bias=bias)  # for f_n
        self.W_fh = nn.Linear(hidden_size, hidden_size, bias=False)  # for f_n
        self.W_gi = nn.Linear(self.num_vd_units*2, hidden_size, bias=bias)  # for g_n
        self.W_gh = nn.Linear(hidden_size, hidden_size, bias=False)  # for g_n
        

        # Output Layer parameters
        self.W_out_I = nn.Linear(hidden_size, 1, bias=bias)
        self.W_out_Q = nn.Linear(hidden_size, 1, bias=bias)

        self.reset_parameters()

    def vd_module(self, I, Q):
        # Vector Demodulator implementation
        magnitude = torch.sqrt(torch.pow(I, 2) + torch.pow(Q, 2))
        # Add epsilon to prevent division by zero
        epsilon = 1e-8
        magnitude = magnitude + epsilon
        magnitude_squared = magnitude**2
        sin_theta = Q / magnitude
        cos_theta = I / magnitude
        return magnitude, magnitude_squared, sin_theta, cos_theta

    def pr_block(self, h, sin_theta, cos_theta, hidden_size, vd_units):
        # Phase Rotation Block as shown in Fig. 10
        if vd_units >= hidden_size:
            cos_theta = cos_theta[:,:,:hidden_size]
            sin_theta = sin_theta[:,:,:hidden_size]
        elif hidden_size > vd_units and hidden_size <= vd_units * 2:
            cos_theta = torch.cat([cos_theta, cos_theta[:,:,:hidden_size-vd_units]], dim=-1)
            sin_theta = torch.cat([sin_theta, sin_theta[:,:,:hidden_size-vd_units]], dim=-1)
        elif hidden_size > vd_units * 2:
            cos_theta = torch.cat([cos_theta, cos_theta,cos_theta[:,:,:hidden_size-2*vd_units]], dim=-1)
            sin_theta = torch.cat([sin_theta, sin_theta,sin_theta[:,:,:hidden_size-2*vd_units]], dim=-1)
        return h * cos_theta, h * sin_theta

    def forward(self, x, h_0):
        # x shape: (batch_size, seq_len, 2)
        # h_0 shape: (num_layers, batch_size, hidden_size) or (batch_size, hidden_size)
        
        batch_size, seq_len, _ = x.shape
        
        # Handle h_0 shape - extract first layer if multiple layers provided
        if h_0 is None:
            h_n = self.init_hidden(batch_size, x.device)
        elif h_0.dim() == 3:  # (num_layers, batch_size, hidden_size)
            h_n = h_0[0]  # Take first layer
        else:  # (batch_size, hidden_size)
            h_n = h_0
            
        h_seq = []

        # Feature Extraction
        feature_size = x.size(2)

        # Split a frame into memory windows
        # zero_pad = torch.zeros((batch_size, self.window_size - 1, feature_size))
        pad = torch.zeros_like(x[:, -(self.window_size - 1):, :]) # zero padding
        x = torch.cat((pad, x), dim=1)
        windows = x.unfold(dimension=1, size=self.window_size, step=1).transpose(2, 3)
        windows = torch.unsqueeze(windows, dim=2)  # Dim: (batch_size, n_windows, 1, window_size, feature_size)
        windows = windows.contiguous().view(-1, seq_len, self.window_size, feature_size)

        # FIR Filters Layer
        I_fir = self.fir_I(windows[:,:,:,0])-self.fir_Q(windows[:,:,:,1])
        I_fir = I_fir.contiguous().view(-1, seq_len, self.num_vd_units)
        Q_fir = self.fir_Q(windows[:,:,:,0])+self.fir_I(windows[:,:,:,1])
        Q_fir = Q_fir.contiguous().view(-1, seq_len, self.num_vd_units)
        magnitude, magnitude_squared, sin_theta, cos_theta = self.vd_module(I_fir, Q_fir)
        L = torch.stack([magnitude, magnitude_squared], dim=2)
        L = L.view(-1, seq_len, self.num_vd_units*2)
        for t in range(seq_len):
            L_n = L[:, t, :]
    
            # Recursive Topology Layer
            f_n = torch.sigmoid(self.W_fi(L_n)+self.W_fh(h_n))
            g_n = torch.tanh(self.W_gi(L_n)+self.W_gh(h_n))
            h_n = f_n * h_n + (1 - f_n) * g_n
            

            h_seq.append(h_n)

        h_seq = torch.stack(h_seq, dim=1)
        h_seq = h_seq.view(-1, seq_len, self.hidden_size)
        I_rot, Q_rot = self.pr_block(h_seq, sin_theta, cos_theta, self.hidden_size, self.num_vd_units)
        out_I = self.W_out_I(I_rot)-self.W_out_Q(Q_rot)
        out_Q = self.W_out_Q(Q_rot)+self.W_out_I(I_rot)
        out = torch.cat([out_I, out_Q], dim=-1)
        return out

    def reset_parameters(self):
        # Initialize FIR filter layers with small weights
        for module in [self.fir_I, self.fir_Q]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain for FIR filters
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        # Initialize input-to-hidden weights with Xavier uniform
        for module in [self.W_fi, self.W_gi]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        # Initialize hidden-to-hidden weights with orthogonal initialization (better for RNNs)
        for module in [self.W_fh, self.W_gh]:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        # Initialize output layers with Xavier uniform
        for module in [self.W_out_I, self.W_out_Q]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def init_hidden(self, batch_size, device):
        """Initialize hidden state with proper shape"""
        return torch.zeros(batch_size, self.hidden_size, device=device)