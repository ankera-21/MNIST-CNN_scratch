import torch
import torch.nn as nn
from torch.nn import Module


class CustomDropout(Module):
    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, input):
        if self.training:
            # Generate a binary mask with `p` probability of being 0
            mask = (torch.rand_like(input) >= self.p).to(input.dtype)
            return input * mask / (1 - self.p)
        return input

  

if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 0
    torch.manual_seed(seed)

    x = torch.randn(1, 3, 5, 5)
    print(x.shape)
    print(x)
    # Custom dropout
    dropout = CustomDropout(p=0.5)
    dropout.train()  # Enable training mode
    dropout_output = dropout(x)
    print(dropout_output.shape)
    print(dropout_output)
    
    