import torch
import torch.nn.functional as F
import torch.nn as nn

class TCNN(nn.Module):
    def __init__(self, hidden_channels):
        super(TCNN, self).__init__()
        self.in_channels = 6
        self.hidden_channels = hidden_channels
        self.out_channels = 2
        self.kernel_size = 5
        self.dilation = 1
        self.stride = 1 
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

            nn.Conv1d(self.hidden_channels, self.out_channels, kernel_size=1, bias=False),
            )
    
    def count_flops(self, input_shape):
        """
        Calculate FLOPs for TCNN model per timestep.
        
        Args:
            input_shape (tuple): Input tensor shape (batch_size, seq_len, 2)
            
        Returns:
            int: Total FLOPs per timestep
        """
        batch_size, seq_len, _ = input_shape
        
        # 1. Input processing (feature extraction)
        # i_x, q_x extraction - no FLOPs
        # amp2 = i_x^2 + q_x^2
        amp2_ops = 3  # 2 multiplications + 1 addition
        # amp = sqrt(amp2)
        amp_ops = 1
        # amp3 = amp^3
        amp3_ops = 2  # 2 multiplications
        # cos = i_x / amp, sin = q_x / amp
        trig_ops = 2  # 2 divisions
        # Total input processing
        input_processing = amp2_ops + amp_ops + amp3_ops + trig_ops
        
        # 2. Convolutional layers
        # Conv1d(6 -> hidden_channels, kernel_size=1)
        conv1_ops = 6 * self.hidden_channels * 1
        
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
        
        # 3. Residual connection (addition)
        residual_ops = self.out_channels
        
        total_flops = input_processing + conv1_ops + hardswish1_ops + dw_conv_ops + conv_final_ops + residual_ops
        
        return total_flops
            
    def forward(self, x, h_0):
        # Feature Extraction
        i_x = torch.unsqueeze(x[..., 0], dim=-1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        cos = i_x / amp
        sin = q_x / amp
        input = torch.cat((i_x, q_x), dim=-1)
        x = torch.cat((i_x, q_x, amp, amp3, sin, cos), dim=-1)
        x_1 = x.transpose(1,2)
        out = self.network(x_1) 
        out = out.transpose(1,2)
        return out + input

