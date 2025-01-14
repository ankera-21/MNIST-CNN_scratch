import torch
import torch.nn as nn
import math

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(CustomConv2d, self).__init__()
        # Initialize kernel with proper parameter registration
        self.kernel = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(out_channels, in_channels, kernel_size, kernel_size), a=math.sqrt(5)))
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None  # Initialize bias if needed
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        # Input shape: (batch_size, in_channels, height, width)
        B, C, H, W = x.shape

        # Calculate output dimensions
        self.output_height = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_width = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Initialize output matrix
        self.output_matrix = torch.zeros((B, self.out_channels, self.output_height, self.output_width), device=x.device)

        # Pad the input
        self.padded_input = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant')

        # Perform convolution
        for i in range(self.output_height):
            for j in range(self.output_width):
                start_h = i * self.stride
                end_h = start_h + self.kernel_size
                start_w = j * self.stride
                end_w = start_w + self.kernel_size

                # Extract the input slice for this region
                input_slice = self.padded_input[:, :, start_h:end_h, start_w:end_w]

                # Perform the convolution manually by multiplying the input slice with the kernel
                for out_channel in range(self.out_channels):
                    self.output_matrix[:, out_channel, i, j] = torch.sum(
                        input_slice * self.kernel[out_channel:out_channel+1, :, :, :], 
                        dim=(1, 2, 3)
                    )
                    
                    # Add bias if it exists
                    if self.bias is not None:
                        self.output_matrix[:, out_channel, i, j] += self.bias[out_channel]

        return self.output_matrix

    
# custom_model.py
if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Create a random input tensor
    x = torch.randn(10, 8, 30, 30)  # (batch_size, in_channels, height, width)

    # Initialize your custom convolution
    custom_conv = CustomConv2d(8, 16, 3)
    custom_output = custom_conv(x)

    # Initialize PyTorch's Conv2d
    pytorch_conv = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
    pytorch_output = pytorch_conv(x)

    # Check if the outputs match
    print("Custom Conv2d Output Shape:", custom_output.shape)  # Should be (10, 16, 38, 38)
    print("PyTorch Conv2d Output Shape:", pytorch_output.shape)  # Should also be (10, 16, 38, 38)

    # Compare the outputs
    output_match = torch.allclose(custom_output, pytorch_output.detach(), atol=1e-6)
    print("Do the outputs match?", output_match)
    print(torch.max(custom_output-pytorch_output))
    custom_conv = CustomConv2d(8, 16, 3)
    print("===")
    print(list(custom_conv.parameters()))  # This should show the kernel and bias parameters.
