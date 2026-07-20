import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class ELM(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, **kwargs):  
        super(ELM, self).__init__()  
        self.input_size = input_size  
        self.hidden_size = hidden_size  
        self.output_size = output_size  
          
        # Random input weights (fixed, not trainable)  
        self.input_weights = nn.Parameter(  
            torch.randn(input_size, hidden_size) * 0.1,   
            requires_grad=False  
        )  
        # Random bias (fixed, not trainable)  
        self.input_bias = nn.Parameter(  
            torch.randn(hidden_size) * 0.1,   
            requires_grad=False  
        )  
          
        # Output weights (trainable for gradient-based compatibility)  
        self.output_weights = nn.Parameter(  
            torch.randn(hidden_size, output_size) * 0.01  
        )  
          
    def forward(self, x, h_0=None):  
        # x shape: (batch, time, input_size)  
        batch_size, seq_len, _ = x.shape  
          
        # Reshape for processing  
        x_flat = x.reshape(-1, self.input_size)  
          
        # Hidden layer: H = activation(X * W_in + b)  
        hidden = torch.matmul(x_flat, self.input_weights) + self.input_bias  
        hidden = torch.sigmoid(hidden)  # Sigmoid activation  
          
        # Output layer: Y = H * W_out  
        output_flat = torch.matmul(hidden, self.output_weights)  
          
        # Reshape back to (batch, time, output_size)  
        output = output_flat.reshape(batch_size, seq_len, self.output_size)  
          
        return output  
      
    def reset_parameters(self):  
        # Re-initialize random weights  
        nn.init.normal_(self.input_weights, mean=0.0, std=0.1)  
        nn.init.normal_(self.input_bias, mean=0.0, std=0.1)  
        nn.init.normal_(self.output_weights, mean=0.0, std=0.01)
