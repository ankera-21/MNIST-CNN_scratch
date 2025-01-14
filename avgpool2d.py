import torch
import torch.nn as nn

class CustomAvgPool2d(nn.Module):
    """
    Custom implementation of AvgPool2d with stride and padding
    """
    def __init__(self, kernel_size, stride=None, padding=0, count_include_pad=True):
        super(CustomAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.count_include_pad = count_include_pad

    def forward(self, input):
        B, C, H, W = input.shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        # Calculate output dimensions
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding
        output_height = (H_padded - kernel_size) // stride + 1
        output_width = (W_padded - kernel_size) // stride + 1

        # Initialize output tensor
        output = torch.zeros(B, C, output_height, output_width, device=input.device)

        # Apply padding to the input
        input_padded = nn.functional.pad(input, (padding, padding, padding, padding))

        for i in range(output_height):
            for j in range(output_width):
                # Define the region to pool
                h_start = i * stride
                h_end = h_start + kernel_size
                w_start = j * stride
                w_end = w_start + kernel_size
                
                # Perform average pooling
                pooled_region = input_padded[:, :, h_start:h_end, w_start:w_end]
                
                # Calculate the mean, ensuring the output shape matches
                if self.count_include_pad:
                    output[:, :, i, j] = pooled_region.mean(dim=[2, 3])
                else:
                    # If not including padding, we need to adjust the mean calculation
                    valid_elements = (pooled_region != 0).sum(dim=[2, 3], keepdim=True)
                    output[:, :, i, j] = pooled_region.sum(dim=[2, 3]) / valid_elements.clamp(min=1)

        return output

if __name__ == "__main__":
    input = torch.randn(1, 3, 28, 28)
    avgpool2d = CustomAvgPool2d(kernel_size=28, stride=1, padding=0)
    output = avgpool2d(input)
    print(output.shape)  # [B, C, H, W]
