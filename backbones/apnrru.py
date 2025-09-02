import torch
from torch import nn


class RRU(nn.Module):
    def __init__(self, hidden_size,window_size, bias=True):
        super(RRU, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_A = 3
        self.num_fir_filters = 3
        self.hidden_node = 16

        # Deep cell structure components
        self.W_u = nn.Linear(hidden_size*2+ self.hidden_size_A+self.num_fir_filters*2+2, self.hidden_node, bias=bias)  # For input concatenation
        self.W_h = nn.Linear(self.hidden_node, hidden_size*2+ self.hidden_size_A, bias=bias)  # First tanh layer
        
        # C and Z are scalar parameters between 0 and 1
        self.C = nn.Parameter(torch.rand(1))  # Scalar parameter
        self.Z = nn.Parameter(torch.zeros(1,2*hidden_size+ self.hidden_size_A))  # Scalar parameter
        
    def forward(self, x, h_prev, h_A_prev):
        # Concatenate input features
        u = torch.cat([x, h_prev,h_A_prev], dim=-1)
        h_new = torch.cat([h_prev,h_A_prev], dim=-1)
        
        # Deep cell computations
        v = torch.tanh(self.W_u(u))
        v = torch.tanh(self.W_h(v))
        v = torch.sigmoid(self.C * h_new) + self.Z * v

        
        return v


class APNRRU(nn.Module):
    def __init__(self, hidden_size, bias=True):
        super(APNRRU, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_A = 3
        self.window_size = 16  # M in the figure
        self.num_fir_filters = 3
        self.hidden_node = 16
        
        # FIR Filters Layer
        self.fir_I = nn.Linear(self.window_size, self.num_fir_filters, bias=False)  # b_{p,0} and b_{p,1}
        self.fir_Q = nn.Linear(self.window_size, self.num_fir_filters, bias=False)  # b_{p,0} and b_{p,1}

        # RRU Cell
        self.rru = RRU(hidden_size, self.window_size, bias)
        
        # Output layer
        self.output_layer_I = nn.Linear(hidden_size, 1, bias=False)
        self.output_layer_Q = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, x, h_0):
        raw_x = x
        # x shape: (batch_size, seq_len, 2)
        batch_size, seq_len, _ = x.shape
        
        # Initialize states
        h_I = torch.zeros(batch_size, self.hidden_size,device=x.device)  # I component of hidden state
        h_Q = torch.zeros(batch_size, self.hidden_size,device=x.device)  # Q component of hidden state
        h_A = torch.zeros(batch_size, self.hidden_size_A,device=x.device)  # Envelope states
        outputs = []


        # Feature Extraction
        feature_size = x.size(2)

        # Split a frame into memory windows
        # zero_pad = torch.zeros((batch_size, self.window_size - 1, feature_size))
        pad = torch.zeros_like(x[:, -(self.window_size - 1):, :])
        x = torch.cat((pad, x), dim=1)
        windows = x.unfold(dimension=1, size=self.window_size, step=1).transpose(2, 3)
        windows = torch.unsqueeze(windows, dim=2)  # Dim: (batch_size, n_windows, 1, window_size, feature_size)
        windows = windows.contiguous().view(-1, seq_len, self.window_size, feature_size)


        last_I = windows[:,:,-1,0]  # (batch_size, seq_len)
        last_Q = windows[:,:,-1,1]
        last_magnitudes = torch.sqrt(last_I**2 + last_Q**2)

         # Calculate r using only the last step of each window
        r = torch.complex(last_I, -last_Q) / (last_magnitudes)  # r(k) = x(k)/|x(k)|
        
        # Phase normalization for entire windows using the last step's r
        r_real = r.real.unsqueeze(-1)  # Add dimension for broadcasting
        r_imag = r.imag.unsqueeze(-1)

        I_fir = self.fir_I(windows[:,:,:,0])-self.fir_Q(windows[:,:,:,1])
        I_fir = I_fir.contiguous().view(-1, seq_len, self.num_fir_filters,1)
        Q_fir = self.fir_Q(windows[:,:,:,0])+self.fir_I(windows[:,:,:,1])
        Q_fir = Q_fir.contiguous().view(-1, seq_len, self.num_fir_filters,1)
        I_fir = torch.cat((I_fir,raw_x[:,:,0].unsqueeze(-1).unsqueeze(-1)), dim=2)
        Q_fir = torch.cat((Q_fir,raw_x[:,:,1].unsqueeze(-1).unsqueeze(-1)), dim=2)
        IQ_fir = torch.cat((I_fir,Q_fir), dim=-1)
        # Normalize all steps in the window using the same r
        IQ_fir_normalized = torch.zeros_like(IQ_fir)
        IQ_fir_normalized[...,0] = (r_real.repeat(1,1,self.num_fir_filters+1) * IQ_fir[...,0] - r_imag.repeat(1,1,self.num_fir_filters+1) * IQ_fir[...,1])
        IQ_fir_normalized[...,1] = (r_imag.repeat(1,1,self.num_fir_filters+1) * IQ_fir[...,0] + r_real.repeat(1,1,self.num_fir_filters+1) * IQ_fir[...,1])
        
        IQ_fir = IQ_fir_normalized.view(batch_size, seq_len, (self.num_fir_filters+1)*feature_size)
        
        for t in range(seq_len):
            x = IQ_fir[:, t, :]
            h_complex = torch.complex(h_I, h_Q) * r[:, t].unsqueeze(-1)
            h_I, h_Q = h_complex.real, h_complex.imag
            h_prev = torch.cat([h_I,h_Q], dim=-1)
            
            # RRU forward pass
            h_next = self.rru(x, 
                            h_prev,
                            h_A)
            
            # Update states
            h_I_next, h_Q_next = h_next[:, :self.hidden_size].unsqueeze(-1), h_next[:, self.hidden_size:2*self.hidden_size].unsqueeze(-1)  # Complex hidden state
            h_A_next = h_next[:, 2*self.hidden_size:]  # Envelope states
            
            # Phase denormalization
            r_conj = torch.complex(r_real[:, t], -r_imag[:, t])  # r*(k)
            h_complex = torch.complex(h_I_next, h_Q_next)
            h_complex = r_conj * h_complex.squeeze(-1)
            h_I_next, h_Q_next = h_complex.real, h_complex.imag

            h_I, h_Q,h_A = h_I_next, h_Q_next,h_A_next
            
            # Generate final output
            output_I = self.output_layer_I(h_I)-self.output_layer_Q(h_Q)
            output_Q = self.output_layer_Q(h_Q)+self.output_layer_I(h_I)
            output = torch.cat([output_I, output_Q], dim=-1)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def reset_parameters(self):
        # Initialize FIR weights
        nn.init.xavier_uniform_(self.fir_I.weight)
        nn.init.xavier_uniform_(self.fir_Q.weight)
        
        # Initialize RRU parameters
        for module in [self.rru.W_u, self.rru.W_h]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

        
