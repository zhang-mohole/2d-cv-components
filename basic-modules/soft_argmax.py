import torch
import torch.nn as nn
from torch.nn import functional as F

"""
This function assumes an input tensor in shape (batch_size, channel, height, width, depth) 
and returns 3D coordinates in shape (batch_size, channel, 3).

For example, if your network output is (batch_size, 16, 64, 64, 64) voxels, 
then the output is 3D coordinates in shape (batch_size, 16, 3). 
This case is usually seen when you has 16 3D heat-voxels and try to find the locations of maximum.

To apply to 2D cases, just set depth to 1 and grab the first two coordinates. 
For example, your network output is 16 (200, 200) heatmaps 
and you are trying to find 2D locations of maximum on each map. 
You can sent (batch_size, 16, 200, 200, 1) to this function, 
and the output would be (batch_size, 3) and you take the first 2 of the 3. 
This idea can be applied to 1D cases by setting both width and depth to 1.

Of course you can set channel to 1 then you'll get one coordinates for each instance in the batch.

Be careful that, you don't want use coordinates.floor() or coordinates.round() 
to get integer coordinates. Because these operations are not differentiable.
"""

def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim()==5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0)
    indices_kernel = indices_kernel.view((H,W,D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    coords = torch.stack([x,y,z],dim=2)
    return coords

def soft_argmax_2d(features):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W)
    Return: x,y coordinates in shape (batch_size, channel, 2)
    """
    assert features.dim()==4
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    N,C,H,W = features.shape
    soft_max = nn.functional.softmax(features.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(features.shape)
    print(features)
    print(soft_max)
    indices_kernel = torch.arange(start=0,end=H*W).unsqueeze(0)
    indices_kernel = indices_kernel.view((H,W))
    conv = soft_max*indices_kernel.float()
    indices = conv.sum(2).sum(2)
    y = indices%W + 1
    x = (indices/W).floor()%H + 1
    coords = torch.stack([y,x],dim=2)
    return coords

class SoftArgmax2D(nn.Module):
    """
    Creates a module that computes Soft-Argmax 2D of a given input heatmap.
    Returns the index of the maximum 2d coordinates of the give map.
    :param beta: The smoothing parameter.
    :param return_xy: The output order is [x, y]. x for colom, y for raw
    """

    def __init__(self, beta: int = 100, return_xy: bool = False):
        if not 0.0 <= beta:
            raise ValueError(f"Invalid beta: {beta}")
        super().__init__()
        self.beta = beta
        self.return_xy = return_xy

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        :param heatmap: The input heatmap is of size B x N x H x W.
        :return: The index of the maximum 2d coordinates is of size B x N x 2.
        """
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, height, width = heatmap.size()
        device: str = heatmap.device

        softmax: torch.Tensor = F.softmax(
            heatmap.view(batch_size, num_channel, height * width), dim=2
        ).view(batch_size, num_channel, height, width)

        xx, yy = torch.meshgrid(list(map(torch.arange, [height, width])))

        approx_x = (
            softmax.mul(xx.float().to(device))
                .view(batch_size, num_channel, height * width)
                .sum(2)
                .unsqueeze(2)
        )
        approx_y = (
            softmax.mul(yy.float().to(device))
                .view(batch_size, num_channel, height * width)
                .sum(2)
                .unsqueeze(2)
        )

        output = [approx_x, approx_y] if self.return_xy else [approx_y, approx_x]
        output = torch.cat(output, 2)
        return output

if __name__ == "__main__":
	# voxel = torch.randn(1,2,2,3,3) # (batch_size, channel, H, W, depth)
	# coords = soft_argmax(voxel)
    hm = torch.randn(1,2,12,12)
    print(hm)
    coords = soft_argmax_2d(hm)
    print(coords)

    soft2d = SoftArgmax2D()
    coords2 = soft2d(hm)
    print(coords2)