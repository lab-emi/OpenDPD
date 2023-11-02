"""
Reproduced from: https://ieeexplore.ieee.org/document/1703853
"""

import torch
from torch import nn


class GMP(nn.Module):
    def __init__(self, L=13, M=12, K=5):
        super(GMP, self).__init__()
        self.M = M
        self.K = K
        # Number of odd orders
        self.n_odd_order = 0
        for k in range(1, self.K + 1, 2):
            self.n_odd_order += 1
        self.L = L
        self.A_kl = nn.Parameter(torch.Tensor(L, self.n_odd_order))
        self.B_klm = nn.Parameter(torch.Tensor(L, M, self.n_odd_order))
        self.C_klm = nn.Parameter(torch.Tensor(L, M, self.n_odd_order))

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'A_kl' in name or 'B_klm' in name or 'C_klm' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, h_0):
        device = x.device
        batch_size = x.size(0)
        frame_length = x.size(1)

        # Convert input to complex numbers
        x_complex = torch.complex(x[..., 0], x[..., 1])  # Dim: (batch_size, frame_length)

        # First Term
        # Split a frame into windows of size L
        # zero_pad = torch.zeros((batch_size, self.L - 1), device=device)   # Dim: (batch_size, L - 1)
        pad = x_complex[:, -(self.L - 1):]
        x_first_term = torch.cat((pad, x_complex), dim=1)  # Dim: (batch_size, frame_length + L - 1)
        x_first_term = x_first_term.unfold(dimension=1, size=self.L, step=1)  # Dim: (batch_size, frame_length, L)
        x_first_term = x_first_term.contiguous().view(-1, self.L)  # Dim: (batch_size * frame_length, L)
        x_first_terms = []
        for k in range(1, self.K + 1, 2):
            x_first_terms.append(x_first_term * torch.pow(torch.abs(x_first_term), exponent=k))
        x_first_term = torch.stack(x_first_terms, dim=-1)
        expanded_A_kl = self.A_kl.unsqueeze(0).expand(batch_size*frame_length, -1, -1)
        product = expanded_A_kl * x_first_term  # Dim: (batch_size * frame_length, L, n_odd_order)
        product = product.view(batch_size, frame_length, -1)  # Dim: (batch_size, frame_length, L * n_odd_order)
        sum_first_term = torch.sum(product, dim=-1)  # Dim: (batch_size, frame_length)

        # Second Term
        # Further split the frame into windows of size M along the L dimension
        # zero_pad = torch.zeros((batch_size, self.L + self.M - 1), device=device)  # Dim: (batch_size, L + M - 1)
        pad = x_complex[:, -(self.L + self.M - 1):]
        x_second_term = torch.cat((pad, x_complex), dim=1)  # Dim: (batch_size, frame_length + L + M - 1)
        x_second_term = x_second_term.unfold(dimension=1, size=self.L + self.M, step=1)  # Dim: (batch_size, frame_length, L + M)
        x_second_term = x_second_term.unfold(dimension=2, size=self.M+1,
                                             step=1)  # Dim: (batch_size, frame_length, L, M + 1)
        x_second_term = x_second_term.contiguous().view(-1, self.L, self.M+1)  # Dim: (batch_size * frame_length, L, M)
        x_second_terms = []
        for k in range(1, self.K + 1, 2):
            x_second_terms.append(x_second_term[..., -1].unsqueeze(-1) * torch.pow(torch.abs(x_second_term[..., :-1]), exponent=k))
        x_second_term = torch.stack(x_second_terms, dim=-1)
        expanded_B_klm = self.B_klm.unsqueeze(0).expand(batch_size * frame_length, -1, -1, -1)
        product = expanded_B_klm * x_second_term
        product = product.view(batch_size, frame_length, -1)  # Dim: (batch_size, frame_length, L * n_odd_order)
        sum_second_term = torch.sum(product, dim=-1)  # Dim: (batch_size, frame_length)

        # Third Term
        #Further split the frame into windows of size M along the L dimension
        # zero_pad = torch.zeros((batch_size, self.L + self.M - 1), device=device)  # Dim: (batch_size, L + M - 1)
        pad = x_complex[:, :self.L + self.M - 1]
        x_third_term = torch.cat((x_complex, pad), dim=1)  # Dim: (batch_size, frame_length + L + M - 1)
        x_third_term = x_third_term.unfold(dimension=1, size=self.L + self.M,
                                             step=1)  # Dim: (batch_size, frame_length, L + M)
        x_third_term = x_third_term.unfold(dimension=2, size=self.M + 1,
                                             step=1)  # Dim: (batch_size, frame_length, L, M + 1)
        x_third_term = x_third_term.contiguous().view(-1, self.L,
                                                        self.M + 1)  # Dim: (batch_size * frame_length, L, M)
        x_third_terms = []
        for k in range(1, self.K + 1, 2):
            x_third_terms.append(
                x_third_term[..., -1].unsqueeze(-1) * torch.pow(torch.abs(x_third_term[..., :-1]), exponent=k))
        x_third_term = torch.stack(x_third_terms, dim=-1)
        expanded_C_klm = self.C_klm.unsqueeze(0).expand(batch_size * frame_length, -1, -1, -1)
        product = expanded_C_klm * x_third_term
        product = product.view(batch_size, frame_length, -1)  # Dim: (batch_size, frame_length, L * n_odd_order)
        sum_third_term = torch.sum(product, dim=-1)  # Dim: (batch_size, frame_length)

        # Merge Terms
        sum_terms = sum_first_term + sum_second_term + sum_third_term

        # Arrange Outputs
        out = torch.zeros((batch_size, frame_length, 2), device=device)
        out[:, :, 0] = torch.real(sum_terms)
        out[:, :, 1] = torch.imag(sum_terms)
        return out
