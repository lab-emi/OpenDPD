import torch
import torch.nn.functional as F


class INT_Conv2D(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, weight_quantizer, act_quantizer):
        super(INT_Conv2D, self).__init__(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode
        )

        self.weight = torch.nn.Parameter(m.weight.detach())

        self.weight_quantizer = weight_quantizer
        self.weight_quantizer.init_step_size(m.weight)

        self.act_quantizer = act_quantizer

            
        # save the parameters 
        self.register_buffer('n_bits_w', torch.Tensor([self.weight_quantizer.bits]))
        self.register_buffer('n_bits_a', torch.Tensor([self.act_quantizer.bits]))
        
    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        # self.register_buffer('quantized_weight', quantized_weight)
        
        quantized_act = self.act_quantizer(x)
        # self.register_buffer('quantized_act', quantized_act)

        # quantized_act = x

        return F.conv2d(quantized_act, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class INT_Linear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Conv2d, weight_quantizer, act_quantizer):
        super(INT_Linear, self).__init__(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=True if m.bias is not None else False)

        self.weight = torch.nn.Parameter(m.weight.detach())        
        self.weight_quantizer = weight_quantizer
        self.weight_quantizer.init_step_size(m.weight)
        self.act_quantizer = act_quantizer
        
        # save the parameters 
        self.register_buffer('n_bits_w', torch.Tensor([self.weight_quantizer.bits]))
        self.register_buffer('n_bits_a', torch.Tensor([self.act_quantizer.bits]))
        

    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        # self.register_buffer('quantized_weight', quantized_weight)

        quantized_act = self.act_quantizer(x)
        # self.register_buffer('quantized_act', quantized_act)

        return F.linear(quantized_act, quantized_weight, self.bias)
