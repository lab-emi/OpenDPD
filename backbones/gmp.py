import torch
from torch import nn


#
# class GMP(nn.Module):
#     def __init__(self, memory_depth, degree):
#         super(GMP, self).__init__()
#         self.filter_length = memory_depth + memory_depth * memory_depth * (degree)
#         self.WI = torch.Tensor(self.filter_length, 1)
#         self.WQ = torch.Tensor(self.filter_length, 1)
#         self.W = nn.Parameter(torch.complex(self.WI, self.WQ))
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         seq_len = x.size(1)
#         out = torch.zeros(batch_size, seq_len, 2)
#         output = torch.squeeze(torch.matmul(x, self.W))
#         real = torch.real(output)
#         out[:, :, 0] = torch.real(output)
#         out[:, :, 1] = torch.imag(output)
#         return out
#
#
# def data_prepare1(X, y, frame_length, degree):
#     Input = []
#     Output = []
#     for k in range(X.shape[0]):
#         X1 = np.zeros((X.shape[1] + frame_length, 2))
#         X1[:, 0] = np.concatenate((X[k, :frame_length, 0], X[k, :, 0]))
#         X1[:, 1] = np.concatenate((X[k, :frame_length:, 1], X[k, :, 1]))
#         X1 = torch.Tensor(X1)
#         y = torch.Tensor(y)
#         Complex_In = torch.complex(X1[:, 0], X1[:, 1])
#         Complex_Out = torch.complex(y[k, :, 0], y[k, :, 1])
#         ulength = len(Complex_In) - frame_length
#         Input_matrix = torch.complex(torch.zeros(ulength, frame_length),
#                                      torch.zeros(ulength, frame_length))
#         degree_matrix = torch.complex(torch.zeros(ulength - frame_length, frame_length * frame_length * degree),
#                                       torch.zeros(ulength - frame_length, frame_length * frame_length * degree))
#         for i in range(ulength):
#             Input_matrix[i, :] = Complex_In[i:i + frame_length]
#         for j in range(1, degree + 1):
#             for h in range(frame_length):
#                 degree_matrix[:,
#                 (j - 1) * frame_length * frame_length + h * frame_length:(j - 1) * frame_length * frame_length + (
#                         h + 1) * frame_length] = Input_matrix[:ulength - frame_length, :] * torch.pow(
#                     abs(Input_matrix[h:h + ulength - frame_length, :]), j)
#         Input_matrix = torch.cat((Input_matrix[:ulength - frame_length], degree_matrix), dim=1)
#         b_output = np.array(Complex_Out)
#         b_input = np.array(Input_matrix)
#         Input.append(b_input)
#         Output.append(b_output)
#
#     return Input, Output
#
#
# def data_prepare1(X, y, frame_length, degree):
#     Input = []
#     Output = []
#     for k in range(X.shape[0]):
#         X1 = np.zeros((X.shape[1] + frame_length, 2))
#         X1[:, 0] = np.concatenate((X[k, :frame_length, 0], X[k, :, 0]))
#         X1[:, 1] = np.concatenate((X[k, :frame_length:, 1], X[k, :, 1]))
#         X1 = torch.Tensor(X1)
#         y = torch.Tensor(y)
#         Complex_In = torch.complex(X1[:, 0], X1[:, 1])
#         Complex_Out = torch.complex(y[k, :, 0], y[k, :, 1])
#         ulength = len(Complex_In) - frame_length
#         Input_matrix = torch.complex(torch.zeros(ulength, frame_length),
#                                      torch.zeros(ulength, frame_length))
#         degree_matrix = torch.complex(torch.zeros(ulength - frame_length, frame_length * frame_length * degree),
#                                       torch.zeros(ulength - frame_length, frame_length * frame_length * degree))
#         for i in range(ulength):
#             Input_matrix[i, :] = Complex_In[i:i + frame_length]
#         for j in range(1, degree + 1):
#             for h in range(frame_length):
#                 degree_matrix[:,
#                 (j - 1) * frame_length * frame_length + h * frame_length:(j - 1) * frame_length * frame_length + (
#                         h + 1) * frame_length] = Input_matrix[:ulength - frame_length, :] * torch.pow(
#                     abs(Input_matrix[h:h + ulength - frame_length, :]), j)
#         Input_matrix = torch.cat((Input_matrix[:ulength - frame_length], degree_matrix), dim=1)
#         b_output = np.array(Complex_Out)
#         b_input = np.array(Input_matrix)
#         Input.append(b_input)
#         Output.append(b_output)
#
#     return Input, Output
#
#
# import torch
# import torch.nn as nn
#

class GMP(nn.Module):
    def __init__(self, K=4, L=10, M=10):
        """
        A, B, C: sets defining the number of coefficients for each term
        K: memory depth
        """
        super(GMP, self).__init__()

        # Assuming A, B, C are lists or sets
        self.K = K
        self.L = L
        self.M = M

        # Defining parameters a, b, and c
        self.a = nn.Parameter(torch.randn(K, L))
        self.b = nn.Parameter(torch.randn(K, L, M))
        self.c = nn.Parameter(torch.randn(K, L, M))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        time_slice = slice(self.L + self.M, None)
        # x = input[:, time_slice, :]
        x = torch.complex(x[..., 0], x[..., 1])
        y = torch.zeros_like(x)
        out = torch.zeros((batch_size, seq_len, 2), device=x.device)

        for n in range(self.L + self.M, seq_len):
            for k in range(self.K):
                # First term
                y[:, n] += torch.sum(self.a[k, :] * x[:, n-(self.L-1):n+1] * torch.pow(x[:, n-(self.L-1):n+1].abs(), k))


        for n in range(self.L + self.M, seq_len):
            for k in range(self.K):
                for l in range(self.L):
                    for m in range(0, self.M):
                        # Second term
                        y[:, n] += self.b[k, l, m] * x[:, n - l] * torch.pow(x[:, n - l - (m+1)].abs(), (k+1))

        out[..., 0] = torch.real(y)
        out[..., 1] = torch.imag(y)
        return out, time_slice
