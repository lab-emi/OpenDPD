import torch
import torch.nn.functional as F
import torch.nn as nn

class NeuralTX(nn.Module):
    def __init__(self, hidden_channels):
        super(NeuralTX, self).__init__()
        self.in_channels = 4
        self.hidden_channels = hidden_channels
        self.out_channels = 2
        self.kernel_size = 5
        self.dilation = 1
        self.stride = 1 
        self.window_size = 5
        self.bias = False


        self.conv_I = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.window_size,bias=False,padding=2)
        self.conv_Q = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.window_size,bias=False,padding=2)

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=1),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size,stride=self.stride, 
                      padding=(self.kernel_size-3)*self.dilation, dilation=self.dilation, groups=self.hidden_channels, bias=False),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size,stride=self.stride, 
                      padding=(self.kernel_size-3)*self.dilation*2, dilation=self.dilation*2, groups=self.hidden_channels, bias=False),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size,stride=self.stride, 
                      padding=(self.kernel_size-3)*self.dilation*4, dilation=self.dilation*4, groups=self.hidden_channels, bias=False),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=self.stride,
                      padding=(self.kernel_size-3)*self.dilation*8, dilation=self.dilation*8, groups=self.hidden_channels, bias=False),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.out_channels, kernel_size=1, bias=False)
            )
        self.IQ_match = nn.Linear(in_features=2, out_features=self.out_channels, bias=False)
        self.reset_parameters()
    def reset_parameters(self):
        for module in [self.conv_I, self.conv_Q]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        for module in [self.network]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        for module in [self.IQ_match]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def count_flops(self, input_shape):
        """
        Calculate FLOPs for TCNN_FD model per timestep.
        
        Args:
            input_shape (tuple): Input tensor shape (batch_size, seq_len, 2)
            
        Returns:
            int: Total FLOPs per timestep
        """
        batch_size, seq_len, _ = input_shape
        

        
        # 2. FIR filtering with conv_I and conv_Q
        # conv_I and conv_Q: Conv1d(1 -> 1, kernel_size=5)
        # i_fir and q_fir calculations involve 2 conv operations each
        # i_fir = conv_I(i_x) - conv_Q(q_x)
        # q_fir = conv_Q(i_x) + conv_I(q_x)
        conv_fir_ops = 4 * self.window_size  # 4 convolutions with kernel_size=5
        fir_add_ops = 2  # 2 additions/subtractions
        
        # 3. Feature extraction
        # amp2 = i_fir^2 + q_fir^2
        amp2_ops = 3  # 2 multiplications + 1 addition
        # amp = sqrt(amp2)
        amp_ops = 1
        # amp3 = amp^3
        amp3_ops = 2  # 2 multiplications
        # Total feature extraction
        feature_ops = amp2_ops + amp_ops + amp3_ops
        
        # 4. Main network convolutional layers
        # Conv1d(4 -> hidden_channels, kernel_size=1)
        conv1_ops = self.in_channels * self.hidden_channels * 1
        
        # Hardswish activations (approximately 4 ops per activation)
        hardswish1_ops = self.hidden_channels * 4
        
        # Depthwise Conv1d layers with different dilations
        # For depthwise conv: ops = kernel_size * hidden_channels
        dw_conv_ops = 0
        for dilation in [1, 2, 4, 8]:
            dw_conv_ops += self.kernel_size * self.hidden_channels
            dw_conv_ops += self.hidden_channels * 4  # Hardswish
        
        # Final Conv1d(hidden_channels -> 2, kernel_size=1)
        conv_final_ops = self.hidden_channels * self.out_channels * 1
        
        # 5. IQ_match linear layer (2 -> 2)
        iq_match_ops = 2 * self.out_channels
        
        # 6. Residual connections (2 additions)
        residual_ops = 2 * self.out_channels
        
        total_flops = (conv_fir_ops + fir_add_ops + feature_ops + 
                      conv1_ops + hardswish1_ops + dw_conv_ops + 
                      conv_final_ops + iq_match_ops + residual_ops)
        
        return total_flops
    
    def forward(self, x, h_0):
        # Feature Extraction
        x = torch.fft.fft(x[:,:,0].unsqueeze(-1)+1j*x[:,:,1].unsqueeze(-1))
        x = torch.cat((x.real,x.imag), dim=-1)
        raw_x = x
        i_x = torch.unsqueeze(x[..., 0], dim=-1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)
        i_fir = self.conv_I(i_x.transpose(1,2)).transpose(1,2)-self.conv_Q(q_x.transpose(1,2)).transpose(1,2)
        q_fir = self.conv_Q(i_x.transpose(1,2)).transpose(1,2)+self.conv_I(q_x.transpose(1,2)).transpose(1,2)
        amp2 = torch.pow(i_fir, 2) + torch.pow(q_fir, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        iq_fir = torch.cat((i_fir, q_fir), dim=-1)
        x = torch.cat((i_fir, q_fir, amp, amp3), dim=-1)
        x = x.transpose(1,2)
        out = self.network(x)
        out = out.transpose(1,2)
        out = out + self.IQ_match(iq_fir)+iq_fir
        return out

