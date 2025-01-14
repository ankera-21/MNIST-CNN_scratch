### PyTorch MNIST CNN Scratch

```python
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

```
---

```python
import torch
import torch.nn as nn
class CustomMaxPool2d(nn.Module):
    """
    Custom implementation of MaxPool2d with stride and padding
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(CustomMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

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
                
                # Perform max pooling
                pooled_region = input_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = pooled_region.max(dim=2)[0].max(dim=2)[0]

        return output



if __name__ == "__main__":
    input = torch.randn(1, 3, 5, 5)
    print(input)
    maxpool2d = CustomMaxPool2d(kernel_size=5, stride=1, padding=0)
    output = maxpool2d(input)

    print(output.shape)  # [B, C, H, W]
    print(output)
```
---
```python
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
    
    
```
---
```python
import torch
import torch.nn as nn

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))
        

if __name__ == "__main__":
    x = torch.randn(1, 3, 5, 5)
    relu = CustomReLU()
    custom_output = relu(x)
    pt_relu = nn.ReLU()
    pytorch_output = pt_relu(x)
    output_match = torch.allclose(custom_output, pytorch_output.detach(), atol=1e-6)
    print(output_match)
    assert output_match, "The outputs do not match"
    
```
---
```
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

```
---
## How to run the code
```bash
python my_model.py
```
---
```bash
torch.Size([1, 10])
 total params 5688
cuda
 total params 5688
CUDA Available? True
EPOCH: 0

  0%|          | 0/59 [00:00<?, ?it/s]
Loss=2.3240163326263428 Batch_id=0 Accuracy=8.50:   0%|          | 0/59 [00:11<?, ?it/s]
Loss=2.3240163326263428 Batch_id=0 Accuracy=8.50:   2%|▏         | 1/59 [00:11<10:49, 11.20s/it]
Loss=2.3255200386047363 Batch_id=1 Accuracy=8.94:   2%|▏         | 1/59 [00:21<10:49, 11.20s/it]
Loss=2.3255200386047363 Batch_id=1 Accuracy=8.94:   3%|▎         | 2/59 [00:21<10:19, 10.87s/it]
Loss=2.31988525390625 Batch_id=2 Accuracy=9.24:   3%|▎         | 2/59 [00:32<10:19, 10.87s/it]  
Loss=2.31988525390625 Batch_id=2 Accuracy=9.24:   5%|▌         | 3/59 [00:32<10:01, 10.75s/it]
Loss=2.3152711391448975 Batch_id=3 Accuracy=9.30:   5%|▌         | 3/59 [00:43<10:01, 10.75s/it]
Loss=2.3152711391448975 Batch_id=3 Accuracy=9.30:   7%|▋         | 4/59 [00:43<09:48, 10.69s/it]
Loss=2.3181633949279785 Batch_id=4 Accuracy=9.22:   7%|▋         | 4/59 [00:53<09:48, 10.69s/it]
Loss=2.3181633949279785 Batch_id=4 Accuracy=9.22:   8%|▊         | 5/59 [00:53<09:35, 10.66s/it]
Loss=2.3097870349884033 Batch_id=5 Accuracy=9.10:   8%|▊         | 5/59 [01:04<09:35, 10.66s/it]
Loss=2.3097870349884033 Batch_id=5 Accuracy=9.10:  10%|█         | 6/59 [01:04<09:25, 10.67s/it]
Loss=2.2999744415283203 Batch_id=6 Accuracy=9.26:  10%|█         | 6/59 [01:14<09:25, 10.67s/it]
Loss=2.2999744415283203 Batch_id=6 Accuracy=9.26:  12%|█▏        | 7/59 [01:14<09:12, 10.63s/it]
Loss=2.3075122833251953 Batch_id=7 Accuracy=9.39:  12%|█▏        | 7/59 [01:25<09:12, 10.63s/it]
Loss=2.3075122833251953 Batch_id=7 Accuracy=9.39:  14%|█▎        | 8/59 [01:25<09:01, 10.61s/it]
Loss=2.2833962440490723 Batch_id=8 Accuracy=9.94:  14%|█▎        | 8/59 [01:36<09:01, 10.61s/it]
Loss=2.2833962440490723 Batch_id=8 Accuracy=9.94:  15%|█▌        | 9/59 [01:36<08:49, 10.60s/it]
Loss=2.2865352630615234 Batch_id=9 Accuracy=10.72:  15%|█▌        | 9/59 [01:46<08:49, 10.60s/it]
Loss=2.2865352630615234 Batch_id=9 Accuracy=10.72:  17%|█▋        | 10/59 [01:46<08:38, 10.58s/it]
Loss=2.2823946475982666 Batch_id=10 Accuracy=11.27:  17%|█▋        | 10/59 [01:57<08:38, 10.58s/it]
Loss=2.2823946475982666 Batch_id=10 Accuracy=11.27:  19%|█▊        | 11/59 [01:57<08:28, 10.60s/it]
Loss=2.274373769760132 Batch_id=11 Accuracy=11.82:  19%|█▊        | 11/59 [02:07<08:28, 10.60s/it] 
Loss=2.274373769760132 Batch_id=11 Accuracy=11.82:  20%|██        | 12/59 [02:07<08:17, 10.59s/it]
Loss=2.2665810585021973 Batch_id=12 Accuracy=12.46:  20%|██        | 12/59 [02:18<08:17, 10.59s/it]
Loss=2.2665810585021973 Batch_id=12 Accuracy=12.46:  22%|██▏       | 13/59 [02:18<08:06, 10.58s/it]
Loss=2.269216299057007 Batch_id=13 Accuracy=12.94:  22%|██▏       | 13/59 [02:28<08:06, 10.58s/it] 
Loss=2.269216299057007 Batch_id=13 Accuracy=12.94:  24%|██▎       | 14/59 [02:28<07:56, 10.58s/it]
Loss=2.2499403953552246 Batch_id=14 Accuracy=13.48:  24%|██▎       | 14/59 [02:39<07:56, 10.58s/it]
Loss=2.2499403953552246 Batch_id=14 Accuracy=13.48:  25%|██▌       | 15/59 [02:39<07:46, 10.59s/it]
Loss=2.2502145767211914 Batch_id=15 Accuracy=13.82:  25%|██▌       | 15/59 [02:50<07:46, 10.59s/it]
Loss=2.2502145767211914 Batch_id=15 Accuracy=13.82:  27%|██▋       | 16/59 [02:50<07:35, 10.58s/it]
Loss=2.259204149246216 Batch_id=16 Accuracy=14.03:  27%|██▋       | 16/59 [03:00<07:35, 10.58s/it] 
Loss=2.259204149246216 Batch_id=16 Accuracy=14.03:  29%|██▉       | 17/59 [03:00<07:24, 10.58s/it]
Loss=2.2456629276275635 Batch_id=17 Accuracy=14.33:  29%|██▉       | 17/59 [03:11<07:24, 10.58s/it]
Loss=2.2456629276275635 Batch_id=17 Accuracy=14.33:  31%|███       | 18/59 [03:11<07:13, 10.59s/it]
Loss=2.2325878143310547 Batch_id=18 Accuracy=14.79:  31%|███       | 18/59 [03:21<07:13, 10.59s/it]
Loss=2.2325878143310547 Batch_id=18 Accuracy=14.79:  32%|███▏      | 19/59 [03:21<07:02, 10.57s/it]
Loss=2.2368643283843994 Batch_id=19 Accuracy=14.97:  32%|███▏      | 19/59 [03:32<07:02, 10.57s/it]
Loss=2.2368643283843994 Batch_id=19 Accuracy=14.97:  34%|███▍      | 20/59 [03:32<06:53, 10.60s/it]
Loss=2.2192022800445557 Batch_id=20 Accuracy=15.26:  34%|███▍      | 20/59 [03:43<06:53, 10.60s/it]
Loss=2.2192022800445557 Batch_id=20 Accuracy=15.26:  36%|███▌      | 21/59 [03:43<06:41, 10.58s/it]
Loss=2.2331197261810303 Batch_id=21 Accuracy=15.37:  36%|███▌      | 21/59 [03:53<06:41, 10.58s/it]
Loss=2.2331197261810303 Batch_id=21 Accuracy=15.37:  37%|███▋      | 22/59 [03:53<06:31, 10.57s/it]
Loss=2.2242813110351562 Batch_id=22 Accuracy=15.60:  37%|███▋      | 22/59 [04:04<06:31, 10.57s/it]
Loss=2.2242813110351562 Batch_id=22 Accuracy=15.60:  39%|███▉      | 23/59 [04:04<06:20, 10.57s/it]
Loss=2.2109577655792236 Batch_id=23 Accuracy=15.92:  39%|███▉      | 23/59 [04:14<06:20, 10.57s/it]
Loss=2.2109577655792236 Batch_id=23 Accuracy=15.92:  41%|████      | 24/59 [04:14<06:09, 10.56s/it]
Loss=2.2089004516601562 Batch_id=24 Accuracy=16.20:  41%|████      | 24/59 [04:25<06:09, 10.56s/it]
Loss=2.2089004516601562 Batch_id=24 Accuracy=16.20:  42%|████▏     | 25/59 [04:25<06:00, 10.60s/it]
Loss=2.191473960876465 Batch_id=25 Accuracy=16.55:  42%|████▏     | 25/59 [04:36<06:00, 10.60s/it] 
Loss=2.191473960876465 Batch_id=25 Accuracy=16.55:  44%|████▍     | 26/59 [04:36<05:50, 10.61s/it]
Loss=2.1976006031036377 Batch_id=26 Accuracy=16.70:  44%|████▍     | 26/59 [04:46<05:50, 10.61s/it]
Loss=2.1976006031036377 Batch_id=26 Accuracy=16.70:  46%|████▌     | 27/59 [04:46<05:39, 10.61s/it]
Loss=2.1987388134002686 Batch_id=27 Accuracy=16.89:  46%|████▌     | 27/59 [04:57<05:39, 10.61s/it]
Loss=2.1987388134002686 Batch_id=27 Accuracy=16.89:  47%|████▋     | 28/59 [04:57<05:29, 10.62s/it]
Loss=2.1856420040130615 Batch_id=28 Accuracy=17.10:  47%|████▋     | 28/59 [05:07<05:29, 10.62s/it]
Loss=2.1856420040130615 Batch_id=28 Accuracy=17.10:  49%|████▉     | 29/59 [05:07<05:18, 10.61s/it]
Loss=2.1874265670776367 Batch_id=29 Accuracy=17.26:  49%|████▉     | 29/59 [05:18<05:18, 10.61s/it]
Loss=2.1874265670776367 Batch_id=29 Accuracy=17.26:  51%|█████     | 30/59 [05:18<05:08, 10.64s/it]
Loss=2.1776397228240967 Batch_id=30 Accuracy=17.40:  51%|█████     | 30/59 [05:29<05:08, 10.64s/it]
Loss=2.1776397228240967 Batch_id=30 Accuracy=17.40:  53%|█████▎    | 31/59 [05:29<04:57, 10.64s/it]
Loss=2.1645305156707764 Batch_id=31 Accuracy=17.66:  53%|█████▎    | 31/59 [05:39<04:57, 10.64s/it]
Loss=2.1645305156707764 Batch_id=31 Accuracy=17.66:  54%|█████▍    | 32/59 [05:39<04:47, 10.64s/it]
Loss=2.1562905311584473 Batch_id=32 Accuracy=17.87:  54%|█████▍    | 32/59 [05:50<04:47, 10.64s/it]
Loss=2.1562905311584473 Batch_id=32 Accuracy=17.87:  56%|█████▌    | 33/59 [05:50<04:36, 10.63s/it]
Loss=2.1680679321289062 Batch_id=33 Accuracy=18.05:  56%|█████▌    | 33/59 [06:01<04:36, 10.63s/it]
Loss=2.1680679321289062 Batch_id=33 Accuracy=18.05:  58%|█████▊    | 34/59 [06:01<04:25, 10.62s/it]
Loss=2.151453733444214 Batch_id=34 Accuracy=18.23:  58%|█████▊    | 34/59 [06:11<04:25, 10.62s/it] 
Loss=2.151453733444214 Batch_id=34 Accuracy=18.23:  59%|█████▉    | 35/59 [06:11<04:15, 10.63s/it]
Loss=2.1507840156555176 Batch_id=35 Accuracy=18.42:  59%|█████▉    | 35/59 [06:22<04:15, 10.63s/it]
Loss=2.1507840156555176 Batch_id=35 Accuracy=18.42:  61%|██████    | 36/59 [06:22<04:04, 10.63s/it]
Loss=2.1479926109313965 Batch_id=36 Accuracy=18.64:  61%|██████    | 36/59 [06:32<04:04, 10.63s/it]
Loss=2.1479926109313965 Batch_id=36 Accuracy=18.64:  63%|██████▎   | 37/59 [06:32<03:53, 10.60s/it]
Loss=2.1339457035064697 Batch_id=37 Accuracy=18.91:  63%|██████▎   | 37/59 [06:43<03:53, 10.60s/it]
Loss=2.1339457035064697 Batch_id=37 Accuracy=18.91:  64%|██████▍   | 38/59 [06:43<03:42, 10.60s/it]
Loss=2.1286163330078125 Batch_id=38 Accuracy=19.11:  64%|██████▍   | 38/59 [06:54<03:42, 10.60s/it]
Loss=2.1286163330078125 Batch_id=38 Accuracy=19.11:  66%|██████▌   | 39/59 [06:54<03:32, 10.62s/it]
Loss=2.135157346725464 Batch_id=39 Accuracy=19.32:  66%|██████▌   | 39/59 [07:04<03:32, 10.62s/it] 
Loss=2.135157346725464 Batch_id=39 Accuracy=19.32:  68%|██████▊   | 40/59 [07:04<03:21, 10.63s/it]
Loss=2.106924533843994 Batch_id=40 Accuracy=19.59:  68%|██████▊   | 40/59 [07:15<03:21, 10.63s/it]
Loss=2.106924533843994 Batch_id=40 Accuracy=19.59:  69%|██████▉   | 41/59 [07:15<03:11, 10.64s/it]
Loss=2.109987258911133 Batch_id=41 Accuracy=19.88:  69%|██████▉   | 41/59 [07:26<03:11, 10.64s/it]
Loss=2.109987258911133 Batch_id=41 Accuracy=19.88:  71%|███████   | 42/59 [07:26<03:01, 10.66s/it]
Loss=2.1144285202026367 Batch_id=42 Accuracy=20.13:  71%|███████   | 42/59 [07:36<03:01, 10.66s/it]
Loss=2.1144285202026367 Batch_id=42 Accuracy=20.13:  73%|███████▎  | 43/59 [07:36<02:50, 10.67s/it]
Loss=2.1055965423583984 Batch_id=43 Accuracy=20.35:  73%|███████▎  | 43/59 [07:47<02:50, 10.67s/it]
Loss=2.1055965423583984 Batch_id=43 Accuracy=20.35:  75%|███████▍  | 44/59 [07:47<02:40, 10.69s/it]
Loss=2.095767021179199 Batch_id=44 Accuracy=20.61:  75%|███████▍  | 44/59 [07:58<02:40, 10.69s/it] 
Loss=2.095767021179199 Batch_id=44 Accuracy=20.61:  76%|███████▋  | 45/59 [07:58<02:29, 10.69s/it]
Loss=2.079336166381836 Batch_id=45 Accuracy=20.89:  76%|███████▋  | 45/59 [08:08<02:29, 10.69s/it]
Loss=2.079336166381836 Batch_id=45 Accuracy=20.89:  78%|███████▊  | 46/59 [08:08<02:19, 10.70s/it]
Loss=2.0837650299072266 Batch_id=46 Accuracy=21.15:  78%|███████▊  | 46/59 [08:19<02:19, 10.70s/it]
Loss=2.0837650299072266 Batch_id=46 Accuracy=21.15:  80%|███████▉  | 47/59 [08:19<02:08, 10.72s/it]
Loss=2.0861656665802 Batch_id=47 Accuracy=21.35:  80%|███████▉  | 47/59 [08:30<02:08, 10.72s/it]   
Loss=2.0861656665802 Batch_id=47 Accuracy=21.35:  81%|████████▏ | 48/59 [08:30<01:57, 10.72s/it]
Loss=2.0756499767303467 Batch_id=48 Accuracy=21.57:  81%|████████▏ | 48/59 [08:41<01:57, 10.72s/it]
Loss=2.0756499767303467 Batch_id=48 Accuracy=21.57:  83%|████████▎ | 49/59 [08:41<01:47, 10.72s/it]
Loss=2.067671537399292 Batch_id=49 Accuracy=21.80:  83%|████████▎ | 49/59 [08:51<01:47, 10.72s/it] 
Loss=2.067671537399292 Batch_id=49 Accuracy=21.80:  85%|████████▍ | 50/59 [08:51<01:36, 10.70s/it]
Loss=2.0525155067443848 Batch_id=50 Accuracy=22.06:  85%|████████▍ | 50/59 [09:02<01:36, 10.70s/it]
Loss=2.0525155067443848 Batch_id=50 Accuracy=22.06:  86%|████████▋ | 51/59 [09:02<01:25, 10.71s/it]
Loss=2.0486435890197754 Batch_id=51 Accuracy=22.29:  86%|████████▋ | 51/59 [09:13<01:25, 10.71s/it]
Loss=2.0486435890197754 Batch_id=51 Accuracy=22.29:  88%|████████▊ | 52/59 [09:13<01:14, 10.71s/it]
Loss=2.0448243618011475 Batch_id=52 Accuracy=22.57:  88%|████████▊ | 52/59 [09:23<01:14, 10.71s/it]
Loss=2.0448243618011475 Batch_id=52 Accuracy=22.57:  90%|████████▉ | 53/59 [09:23<01:04, 10.71s/it]
Loss=2.0332486629486084 Batch_id=53 Accuracy=22.81:  90%|████████▉ | 53/59 [09:34<01:04, 10.71s/it]
Loss=2.0332486629486084 Batch_id=53 Accuracy=22.81:  92%|█████████▏| 54/59 [09:34<00:53, 10.73s/it]
Loss=2.0293171405792236 Batch_id=54 Accuracy=23.06:  92%|█████████▏| 54/59 [09:45<00:53, 10.73s/it]
Loss=2.0293171405792236 Batch_id=54 Accuracy=23.06:  93%|█████████▎| 55/59 [09:45<00:42, 10.72s/it]
Loss=2.022022247314453 Batch_id=55 Accuracy=23.32:  93%|█████████▎| 55/59 [09:56<00:42, 10.72s/it] 
Loss=2.022022247314453 Batch_id=55 Accuracy=23.32:  95%|█████████▍| 56/59 [09:56<00:32, 10.71s/it]
Loss=2.019191265106201 Batch_id=56 Accuracy=23.55:  95%|█████████▍| 56/59 [10:06<00:32, 10.71s/it]
Loss=2.019191265106201 Batch_id=56 Accuracy=23.55:  97%|█████████▋| 57/59 [10:06<00:21, 10.71s/it]
Loss=1.988939642906189 Batch_id=57 Accuracy=23.81:  97%|█████████▋| 57/59 [10:17<00:21, 10.71s/it]
Loss=1.988939642906189 Batch_id=57 Accuracy=23.81:  98%|█████████▊| 58/59 [10:17<00:10, 10.72s/it]
Loss=2.0077741146087646 Batch_id=58 Accuracy=23.95:  98%|█████████▊| 58/59 [10:24<00:10, 10.72s/it]
Loss=2.0077741146087646 Batch_id=58 Accuracy=23.95: 100%|██████████| 59/59 [10:24<00:00,  9.60s/it]
Loss=2.0077741146087646 Batch_id=58 Accuracy=23.95: 100%|██████████| 59/59 [10:24<00:00, 10.59s/it]

Test set: Average loss: 1.9815, Accuracy: 4000/10000 (40.00%)

EPOCH: 1

  0%|          | 0/59 [00:00<?, ?it/s]
Loss=1.9798470735549927 Batch_id=0 Accuracy=38.48:   0%|          | 0/59 [00:11<?, ?it/s]
Loss=1.9798470735549927 Batch_id=0 Accuracy=38.48:   2%|▏         | 1/59 [00:11<11:30, 11.90s/it]
Loss=1.9601304531097412 Batch_id=1 Accuracy=40.48:   2%|▏         | 1/59 [00:22<11:30, 11.90s/it]
Loss=1.9601304531097412 Batch_id=1 Accuracy=40.48:   3%|▎         | 2/59 [00:22<10:37, 11.18s/it]
Loss=1.9739059209823608 Batch_id=2 Accuracy=40.17:   3%|▎         | 2/59 [00:33<10:37, 11.18s/it]
Loss=1.9739059209823608 Batch_id=2 Accuracy=40.17:   5%|▌         | 3/59 [00:33<10:14, 10.97s/it]
Loss=1.9551345109939575 Batch_id=3 Accuracy=39.60:   5%|▌         | 3/59 [00:44<10:14, 10.97s/it]
Loss=1.9551345109939575 Batch_id=3 Accuracy=39.60:   7%|▋         | 4/59 [00:44<10:01, 10.94s/it]
Loss=1.9525521993637085 Batch_id=4 Accuracy=39.75:   7%|▋         | 4/59 [00:54<10:01, 10.94s/it]
Loss=1.9525521993637085 Batch_id=4 Accuracy=39.75:   8%|▊         | 5/59 [00:54<09:46, 10.85s/it]
Loss=1.924075961112976 Batch_id=5 Accuracy=40.38:   8%|▊         | 5/59 [01:05<09:46, 10.85s/it] 
Loss=1.924075961112976 Batch_id=5 Accuracy=40.38:  10%|█         | 6/59 [01:05<09:33, 10.82s/it]
Loss=1.943109393119812 Batch_id=6 Accuracy=40.47:  10%|█         | 6/59 [01:16<09:33, 10.82s/it]
Loss=1.943109393119812 Batch_id=6 Accuracy=40.47:  12%|█▏        | 7/59 [01:16<09:21, 10.79s/it]
Loss=1.9215331077575684 Batch_id=7 Accuracy=40.76:  12%|█▏        | 7/59 [01:27<09:21, 10.79s/it]
Loss=1.9215331077575684 Batch_id=7 Accuracy=40.76:  14%|█▎        | 8/59 [01:27<09:09, 10.78s/it]
Loss=1.9220739603042603 Batch_id=8 Accuracy=40.72:  14%|█▎        | 8/59 [01:37<09:09, 10.78s/it]
Loss=1.9220739603042603 Batch_id=8 Accuracy=40.72:  15%|█▌        | 9/59 [01:37<08:59, 10.80s/it]
Loss=1.9084830284118652 Batch_id=9 Accuracy=40.75:  15%|█▌        | 9/59 [01:48<08:59, 10.80s/it]
Loss=1.9084830284118652 Batch_id=9 Accuracy=40.75:  17%|█▋        | 10/59 [01:48<08:47, 10.77s/it]
Loss=1.9071879386901855 Batch_id=10 Accuracy=40.84:  17%|█▋        | 10/59 [01:59<08:47, 10.77s/it]
Loss=1.9071879386901855 Batch_id=10 Accuracy=40.84:  19%|█▊        | 11/59 [01:59<08:36, 10.75s/it]
Loss=1.899189829826355 Batch_id=11 Accuracy=40.92:  19%|█▊        | 11/59 [02:10<08:36, 10.75s/it] 
Loss=1.899189829826355 Batch_id=11 Accuracy=40.92:  20%|██        | 12/59 [02:10<08:25, 10.76s/it]
Loss=1.8998711109161377 Batch_id=12 Accuracy=40.84:  20%|██        | 12/59 [02:20<08:25, 10.76s/it]
Loss=1.8998711109161377 Batch_id=12 Accuracy=40.84:  22%|██▏       | 13/59 [02:20<08:14, 10.76s/it]
Loss=1.8715107440948486 Batch_id=13 Accuracy=40.97:  22%|██▏       | 13/59 [02:31<08:14, 10.76s/it]
Loss=1.8715107440948486 Batch_id=13 Accuracy=40.97:  24%|██▎       | 14/59 [02:31<08:05, 10.78s/it]
Loss=1.8515372276306152 Batch_id=14 Accuracy=41.07:  24%|██▎       | 14/59 [02:42<08:05, 10.78s/it]
Loss=1.8515372276306152 Batch_id=14 Accuracy=41.07:  25%|██▌       | 15/59 [02:42<07:54, 10.77s/it]
Loss=1.8485926389694214 Batch_id=15 Accuracy=41.18:  25%|██▌       | 15/59 [02:53<07:54, 10.77s/it]
Loss=1.8485926389694214 Batch_id=15 Accuracy=41.18:  27%|██▋       | 16/59 [02:53<07:42, 10.76s/it]
Loss=1.8391273021697998 Batch_id=16 Accuracy=41.37:  27%|██▋       | 16/59 [03:03<07:42, 10.76s/it]
Loss=1.8391273021697998 Batch_id=16 Accuracy=41.37:  29%|██▉       | 17/59 [03:03<07:31, 10.75s/it]
Loss=1.8393844366073608 Batch_id=17 Accuracy=41.50:  29%|██▉       | 17/59 [03:14<07:31, 10.75s/it]
Loss=1.8393844366073608 Batch_id=17 Accuracy=41.50:  31%|███       | 18/59 [03:14<07:21, 10.77s/it]
Loss=1.8109149932861328 Batch_id=18 Accuracy=41.58:  31%|███       | 18/59 [03:25<07:21, 10.77s/it]
Loss=1.8109149932861328 Batch_id=18 Accuracy=41.58:  32%|███▏      | 19/59 [03:25<07:10, 10.77s/it]
Loss=1.8317337036132812 Batch_id=19 Accuracy=41.67:  32%|███▏      | 19/59 [03:36<07:10, 10.77s/it]
Loss=1.8317337036132812 Batch_id=19 Accuracy=41.67:  34%|███▍      | 20/59 [03:36<06:59, 10.77s/it]
Loss=1.7797017097473145 Batch_id=20 Accuracy=41.82:  34%|███▍      | 20/59 [03:47<06:59, 10.77s/it]
Loss=1.7797017097473145 Batch_id=20 Accuracy=41.82:  36%|███▌      | 21/59 [03:47<06:48, 10.76s/it]
Loss=1.796035647392273 Batch_id=21 Accuracy=41.86:  36%|███▌      | 21/59 [03:57<06:48, 10.76s/it] 
Loss=1.796035647392273 Batch_id=21 Accuracy=41.86:  37%|███▋      | 22/59 [03:57<06:38, 10.76s/it]
Loss=1.791022777557373 Batch_id=22 Accuracy=41.88:  37%|███▋      | 22/59 [04:08<06:38, 10.76s/it]
Loss=1.791022777557373 Batch_id=22 Accuracy=41.88:  39%|███▉      | 23/59 [04:08<06:28, 10.78s/it]
Loss=1.7821953296661377 Batch_id=23 Accuracy=41.94:  39%|███▉      | 23/59 [04:19<06:28, 10.78s/it]
Loss=1.7821953296661377 Batch_id=23 Accuracy=41.94:  41%|████      | 24/59 [04:19<06:17, 10.79s/it]
Loss=1.7845449447631836 Batch_id=24 Accuracy=41.98:  41%|████      | 24/59 [04:30<06:17, 10.79s/it]
Loss=1.7845449447631836 Batch_id=24 Accuracy=41.98:  42%|████▏     | 25/59 [04:30<06:06, 10.78s/it]
Loss=1.7402067184448242 Batch_id=25 Accuracy=42.11:  42%|████▏     | 25/59 [04:40<06:06, 10.78s/it]
Loss=1.7402067184448242 Batch_id=25 Accuracy=42.11:  44%|████▍     | 26/59 [04:40<05:55, 10.77s/it]
Loss=1.7733700275421143 Batch_id=26 Accuracy=42.05:  44%|████▍     | 26/59 [04:51<05:55, 10.77s/it]
Loss=1.7733700275421143 Batch_id=26 Accuracy=42.05:  46%|████▌     | 27/59 [04:51<05:44, 10.77s/it]
Loss=1.736456036567688 Batch_id=27 Accuracy=42.12:  46%|████▌     | 27/59 [05:02<05:44, 10.77s/it] 
Loss=1.736456036567688 Batch_id=27 Accuracy=42.12:  47%|████▋     | 28/59 [05:02<05:34, 10.78s/it]
Loss=1.7313567399978638 Batch_id=28 Accuracy=42.21:  47%|████▋     | 28/59 [05:13<05:34, 10.78s/it]
Loss=1.7313567399978638 Batch_id=28 Accuracy=42.21:  49%|████▉     | 29/59 [05:13<05:23, 10.78s/it]
Loss=1.7328543663024902 Batch_id=29 Accuracy=42.31:  49%|████▉     | 29/59 [05:24<05:23, 10.78s/it]
Loss=1.7328543663024902 Batch_id=29 Accuracy=42.31:  51%|█████     | 30/59 [05:24<05:12, 10.77s/it]
Loss=1.7047675848007202 Batch_id=30 Accuracy=42.38:  51%|█████     | 30/59 [05:34<05:12, 10.77s/it]
Loss=1.7047675848007202 Batch_id=30 Accuracy=42.38:  53%|█████▎    | 31/59 [05:34<05:01, 10.77s/it]
Loss=1.7158650159835815 Batch_id=31 Accuracy=42.43:  53%|█████▎    | 31/59 [05:45<05:01, 10.77s/it]
Loss=1.7158650159835815 Batch_id=31 Accuracy=42.43:  54%|█████▍    | 32/59 [05:45<04:50, 10.75s/it]
Loss=1.6903998851776123 Batch_id=32 Accuracy=42.53:  54%|█████▍    | 32/59 [05:56<04:50, 10.75s/it]
Loss=1.6903998851776123 Batch_id=32 Accuracy=42.53:  56%|█████▌    | 33/59 [05:56<04:39, 10.75s/it]
Loss=1.6995079517364502 Batch_id=33 Accuracy=42.65:  56%|█████▌    | 33/59 [06:07<04:39, 10.75s/it]
Loss=1.6995079517364502 Batch_id=33 Accuracy=42.65:  58%|█████▊    | 34/59 [06:07<04:28, 10.75s/it]
Loss=1.6880236864089966 Batch_id=34 Accuracy=42.81:  58%|█████▊    | 34/59 [06:17<04:28, 10.75s/it]
Loss=1.6880236864089966 Batch_id=34 Accuracy=42.81:  59%|█████▉    | 35/59 [06:17<04:18, 10.75s/it]
Loss=1.6792880296707153 Batch_id=35 Accuracy=42.93:  59%|█████▉    | 35/59 [06:28<04:18, 10.75s/it]
Loss=1.6792880296707153 Batch_id=35 Accuracy=42.93:  61%|██████    | 36/59 [06:28<04:07, 10.75s/it]
Loss=1.6871892213821411 Batch_id=36 Accuracy=43.07:  61%|██████    | 36/59 [06:39<04:07, 10.75s/it]
Loss=1.6871892213821411 Batch_id=36 Accuracy=43.07:  63%|██████▎   | 37/59 [06:39<03:56, 10.76s/it]
Loss=1.6609852313995361 Batch_id=37 Accuracy=43.18:  63%|██████▎   | 37/59 [06:50<03:56, 10.76s/it]
Loss=1.6609852313995361 Batch_id=37 Accuracy=43.18:  64%|██████▍   | 38/59 [06:50<03:46, 10.76s/it]
Loss=1.6184970140457153 Batch_id=38 Accuracy=43.40:  64%|██████▍   | 38/59 [07:00<03:46, 10.76s/it]
Loss=1.6184970140457153 Batch_id=38 Accuracy=43.40:  66%|██████▌   | 39/59 [07:00<03:35, 10.77s/it]
Loss=1.625664472579956 Batch_id=39 Accuracy=43.62:  66%|██████▌   | 39/59 [07:11<03:35, 10.77s/it] 
Loss=1.625664472579956 Batch_id=39 Accuracy=43.62:  68%|██████▊   | 40/59 [07:11<03:24, 10.77s/it]
Loss=1.629289984703064 Batch_id=40 Accuracy=43.83:  68%|██████▊   | 40/59 [07:22<03:24, 10.77s/it]
Loss=1.629289984703064 Batch_id=40 Accuracy=43.83:  69%|██████▉   | 41/59 [07:22<03:13, 10.76s/it]
Loss=1.6226139068603516 Batch_id=41 Accuracy=43.98:  69%|██████▉   | 41/59 [07:33<03:13, 10.76s/it]
Loss=1.6226139068603516 Batch_id=41 Accuracy=43.98:  71%|███████   | 42/59 [07:33<03:03, 10.78s/it]
Loss=1.600771188735962 Batch_id=42 Accuracy=44.15:  71%|███████   | 42/59 [07:43<03:03, 10.78s/it] 
Loss=1.600771188735962 Batch_id=42 Accuracy=44.15:  73%|███████▎  | 43/59 [07:43<02:52, 10.75s/it]
Loss=1.6089422702789307 Batch_id=43 Accuracy=44.30:  73%|███████▎  | 43/59 [07:54<02:52, 10.75s/it]
Loss=1.6089422702789307 Batch_id=43 Accuracy=44.30:  75%|███████▍  | 44/59 [07:54<02:40, 10.73s/it]
Loss=1.5665842294692993 Batch_id=44 Accuracy=44.56:  75%|███████▍  | 44/59 [08:05<02:40, 10.73s/it]
Loss=1.5665842294692993 Batch_id=44 Accuracy=44.56:  76%|███████▋  | 45/59 [08:05<02:30, 10.71s/it]
Loss=1.5704786777496338 Batch_id=45 Accuracy=44.77:  76%|███████▋  | 45/59 [08:15<02:30, 10.71s/it]
Loss=1.5704786777496338 Batch_id=45 Accuracy=44.77:  78%|███████▊  | 46/59 [08:15<02:19, 10.70s/it]
Loss=1.5925886631011963 Batch_id=46 Accuracy=44.96:  78%|███████▊  | 46/59 [08:26<02:19, 10.70s/it]
Loss=1.5925886631011963 Batch_id=46 Accuracy=44.96:  80%|███████▉  | 47/59 [08:26<02:08, 10.72s/it]
Loss=1.5589325428009033 Batch_id=47 Accuracy=45.19:  80%|███████▉  | 47/59 [08:37<02:08, 10.72s/it]
Loss=1.5589325428009033 Batch_id=47 Accuracy=45.19:  81%|████████▏ | 48/59 [08:37<01:57, 10.71s/it]
Loss=1.5758146047592163 Batch_id=48 Accuracy=45.38:  81%|████████▏ | 48/59 [08:48<01:57, 10.71s/it]
Loss=1.5758146047592163 Batch_id=48 Accuracy=45.38:  83%|████████▎ | 49/59 [08:48<01:47, 10.72s/it]
Loss=1.5326919555664062 Batch_id=49 Accuracy=45.59:  83%|████████▎ | 49/59 [08:58<01:47, 10.72s/it]
Loss=1.5326919555664062 Batch_id=49 Accuracy=45.59:  85%|████████▍ | 50/59 [08:58<01:36, 10.73s/it]
Loss=1.5137907266616821 Batch_id=50 Accuracy=45.84:  85%|████████▍ | 50/59 [09:09<01:36, 10.73s/it]
Loss=1.5137907266616821 Batch_id=50 Accuracy=45.84:  86%|████████▋ | 51/59 [09:09<01:25, 10.71s/it]
Loss=1.5269973278045654 Batch_id=51 Accuracy=46.04:  86%|████████▋ | 51/59 [09:20<01:25, 10.71s/it]
Loss=1.5269973278045654 Batch_id=51 Accuracy=46.04:  88%|████████▊ | 52/59 [09:20<01:15, 10.73s/it]
Loss=1.5079381465911865 Batch_id=52 Accuracy=46.34:  88%|████████▊ | 52/59 [09:31<01:15, 10.73s/it]
Loss=1.5079381465911865 Batch_id=52 Accuracy=46.34:  90%|████████▉ | 53/59 [09:31<01:04, 10.73s/it]
Loss=1.478003740310669 Batch_id=53 Accuracy=46.63:  90%|████████▉ | 53/59 [09:41<01:04, 10.73s/it] 
Loss=1.478003740310669 Batch_id=53 Accuracy=46.63:  92%|█████████▏| 54/59 [09:41<00:53, 10.72s/it]
Loss=1.5202358961105347 Batch_id=54 Accuracy=46.78:  92%|█████████▏| 54/59 [09:52<00:53, 10.72s/it]
Loss=1.5202358961105347 Batch_id=54 Accuracy=46.78:  93%|█████████▎| 55/59 [09:52<00:42, 10.70s/it]
Loss=1.4690558910369873 Batch_id=55 Accuracy=47.07:  93%|█████████▎| 55/59 [10:03<00:42, 10.70s/it]
Loss=1.4690558910369873 Batch_id=55 Accuracy=47.07:  95%|█████████▍| 56/59 [10:03<00:32, 10.72s/it]
Loss=1.4866260290145874 Batch_id=56 Accuracy=47.28:  95%|█████████▍| 56/59 [10:13<00:32, 10.72s/it]
Loss=1.4866260290145874 Batch_id=56 Accuracy=47.28:  97%|█████████▋| 57/59 [10:13<00:21, 10.72s/it]
Loss=1.4868979454040527 Batch_id=57 Accuracy=47.50:  97%|█████████▋| 57/59 [10:24<00:21, 10.72s/it]
Loss=1.4868979454040527 Batch_id=57 Accuracy=47.50:  98%|█████████▊| 58/59 [10:24<00:10, 10.72s/it]
Loss=1.4352751970291138 Batch_id=58 Accuracy=47.65:  98%|█████████▊| 58/59 [10:31<00:10, 10.72s/it]
Loss=1.4352751970291138 Batch_id=58 Accuracy=47.65: 100%|██████████| 59/59 [10:31<00:00,  9.59s/it]
Loss=1.4352751970291138 Batch_id=58 Accuracy=47.65: 100%|██████████| 59/59 [10:31<00:00, 10.71s/it]

Test set: Average loss: 1.4654, Accuracy: 5614/10000 (56.14%)

EPOCH: 2

  0%|          | 0/59 [00:00<?, ?it/s]
Loss=1.4376388788223267 Batch_id=0 Accuracy=65.04:   0%|          | 0/59 [00:12<?, ?it/s]
Loss=1.4376388788223267 Batch_id=0 Accuracy=65.04:   2%|▏         | 1/59 [00:12<11:57, 12.36s/it]
Loss=1.4340758323669434 Batch_id=1 Accuracy=63.92:   2%|▏         | 1/59 [00:23<11:57, 12.36s/it]
Loss=1.4340758323669434 Batch_id=1 Accuracy=63.92:   3%|▎         | 2/59 [00:23<10:51, 11.43s/it]
Loss=1.4118373394012451 Batch_id=2 Accuracy=64.32:   3%|▎         | 2/59 [00:33<10:51, 11.43s/it]
Loss=1.4118373394012451 Batch_id=2 Accuracy=64.32:   5%|▌         | 3/59 [00:33<10:19, 11.07s/it]
Loss=1.439401388168335 Batch_id=3 Accuracy=64.09:   5%|▌         | 3/59 [00:44<10:19, 11.07s/it] 
Loss=1.439401388168335 Batch_id=3 Accuracy=64.09:   7%|▋         | 4/59 [00:44<10:00, 10.92s/it]
Loss=1.423588752746582 Batch_id=4 Accuracy=64.10:   7%|▋         | 4/59 [00:55<10:00, 10.92s/it]
Loss=1.423588752746582 Batch_id=4 Accuracy=64.10:   8%|▊         | 5/59 [00:55<09:45, 10.84s/it]
Loss=1.3957961797714233 Batch_id=5 Accuracy=64.06:   8%|▊         | 5/59 [01:05<09:45, 10.84s/it]
Loss=1.3957961797714233 Batch_id=5 Accuracy=64.06:  10%|█         | 6/59 [01:05<09:31, 10.78s/it]
Loss=1.3996610641479492 Batch_id=6 Accuracy=64.29:  10%|█         | 6/59 [01:16<09:31, 10.78s/it]
Loss=1.3996610641479492 Batch_id=6 Accuracy=64.29:  12%|█▏        | 7/59 [01:16<09:18, 10.74s/it]
Loss=1.3861979246139526 Batch_id=7 Accuracy=64.62:  12%|█▏        | 7/59 [01:27<09:18, 10.74s/it]
Loss=1.3861979246139526 Batch_id=7 Accuracy=64.62:  14%|█▎        | 8/59 [01:27<09:06, 10.71s/it]
Loss=1.3744516372680664 Batch_id=8 Accuracy=64.90:  14%|█▎        | 8/59 [01:37<09:06, 10.71s/it]
Loss=1.3744516372680664 Batch_id=8 Accuracy=64.90:  15%|█▌        | 9/59 [01:37<08:54, 10.68s/it]
Loss=1.3867626190185547 Batch_id=9 Accuracy=64.73:  15%|█▌        | 9/59 [01:48<08:54, 10.68s/it]
Loss=1.3867626190185547 Batch_id=9 Accuracy=64.73:  17%|█▋        | 10/59 [01:48<08:42, 10.67s/it]
Loss=1.3717405796051025 Batch_id=10 Accuracy=64.65:  17%|█▋        | 10/59 [01:58<08:42, 10.67s/it]
Loss=1.3717405796051025 Batch_id=10 Accuracy=64.65:  19%|█▊        | 11/59 [01:58<08:30, 10.64s/it]
Loss=1.3437176942825317 Batch_id=11 Accuracy=64.74:  19%|█▊        | 11/59 [02:09<08:30, 10.64s/it]
Loss=1.3437176942825317 Batch_id=11 Accuracy=64.74:  20%|██        | 12/59 [02:09<08:20, 10.65s/it]
Loss=1.3333830833435059 Batch_id=12 Accuracy=64.95:  20%|██        | 12/59 [02:20<08:20, 10.65s/it]
Loss=1.3333830833435059 Batch_id=12 Accuracy=64.95:  22%|██▏       | 13/59 [02:20<08:09, 10.64s/it]
Loss=1.3309643268585205 Batch_id=13 Accuracy=65.12:  22%|██▏       | 13/59 [02:30<08:09, 10.64s/it]
Loss=1.3309643268585205 Batch_id=13 Accuracy=65.12:  24%|██▎       | 14/59 [02:30<07:58, 10.64s/it]
Loss=1.3340524435043335 Batch_id=14 Accuracy=65.01:  24%|██▎       | 14/59 [02:41<07:58, 10.64s/it]
Loss=1.3340524435043335 Batch_id=14 Accuracy=65.01:  25%|██▌       | 15/59 [02:41<07:47, 10.63s/it]
Loss=1.3189018964767456 Batch_id=15 Accuracy=65.14:  25%|██▌       | 15/59 [02:52<07:47, 10.63s/it]
Loss=1.3189018964767456 Batch_id=15 Accuracy=65.14:  27%|██▋       | 16/59 [02:52<07:38, 10.65s/it]
Loss=1.2886089086532593 Batch_id=16 Accuracy=65.43:  27%|██▋       | 16/59 [03:02<07:38, 10.65s/it]
Loss=1.2886089086532593 Batch_id=16 Accuracy=65.43:  29%|██▉       | 17/59 [03:02<07:26, 10.63s/it]
Loss=1.3167587518692017 Batch_id=17 Accuracy=65.46:  29%|██▉       | 17/59 [03:13<07:26, 10.63s/it]
Loss=1.3167587518692017 Batch_id=17 Accuracy=65.46:  31%|███       | 18/59 [03:13<07:15, 10.63s/it]
Loss=1.2803207635879517 Batch_id=18 Accuracy=65.68:  31%|███       | 18/59 [03:24<07:15, 10.63s/it]
Loss=1.2803207635879517 Batch_id=18 Accuracy=65.68:  32%|███▏      | 19/59 [03:24<07:04, 10.62s/it]
Loss=1.2723850011825562 Batch_id=19 Accuracy=65.80:  32%|███▏      | 19/59 [03:34<07:04, 10.62s/it]
Loss=1.2723850011825562 Batch_id=19 Accuracy=65.80:  34%|███▍      | 20/59 [03:34<06:53, 10.61s/it]
Loss=1.274620532989502 Batch_id=20 Accuracy=65.89:  34%|███▍      | 20/59 [03:45<06:53, 10.61s/it] 
Loss=1.274620532989502 Batch_id=20 Accuracy=65.89:  36%|███▌      | 21/59 [03:45<06:43, 10.61s/it]
```