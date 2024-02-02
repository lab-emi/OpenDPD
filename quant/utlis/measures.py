import torch


# calculate the similarity between two tensors
def calc_similarity(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    return torch.dot(tensor1, tensor2) / (torch.norm(tensor1) * torch.norm(tensor2))


# calculate the loss of two tensors
def calc_loss(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    return torch.norm(tensor1 - tensor2) / torch.norm(tensor1)

# calculate the identity ratio of two tensors
def calc_identity_ratio(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    
    # if tensor1 - tensor2 is less than 1e-5, then we think they are the same
    return torch.sum(torch.abs(tensor1 - tensor2) < 1e-4) / tensor1.shape[0]
