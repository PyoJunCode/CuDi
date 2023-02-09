import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from kornia.losses import total_variation

from arch.led_arch import LEDCurveNet

class ColorLoss(nn.Module):

    def __init__(self, loss_weight=1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        k = torch.mean(k)

        return self.loss_weight * k

			
class SpatialLoss(nn.Module):

    def __init__(self, loss_weight=1):
        super().__init__()
        self.loss_weight = loss_weight
        
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, x , pred):
        b,c,h,w = x.shape

        org_mean = torch.mean(x,1,keepdim=True)
        enhance_mean = torch.mean(pred,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        # weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = torch.mean(D_left + D_right + D_up +D_down)


        return self.loss_weight * E

class SpaExpLoss(nn.Module):

    def __init__(self, patch_size=16, loss_weight=1):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.loss_weight = loss_weight

    def forward(self, x, pred):

        b,c,h,w = x.shape
        # RGB mean
        x = torch.mean(x, 1, keepdim=True)
        pred = torch.mean(pred, 1, keepdim=True)
        # Pooling
        mean_x = self.pool(x)
        mean_pred = self.pool(pred)

        # Calculate Distance
        distance = torch.mean(torch.pow(mean_pred - mean_x, 2))
        return self.loss_weight * distance


class IlluSmoothLoss(nn.Module):
    def __init__(self,loss_weight=1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        #print(torch.mean(h_tv/count_h + w_tv/count_w))
        #print((h_tv/count_h+w_tv/count_w)/batch_size)
        return self.loss_weight * 2 * (h_tv/count_h+w_tv/count_w)/batch_size
        #return self.loss_weight * torch.mean(h_tv/count_h + w_tv/count_w)

# class IlluSmoothLoss(nn.Module):
#     def __init__(self, loss_weight=1):
#         super().__init__()
#         self.loss_weight = loss_weight

#     def forward(self, x):
#         loss = total_variation(x, reduction="mean")
#         loss = torch.mean(loss)

#         return self.loss_weight * loss

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * self.l1_loss(pred, target)


class StudentLoss(nn.Module):
    def __init__(self, teacher_weight, loss_weight=1.0):
        super().__init__()

        self.loss_weight = loss_weight

        self.l1_loss = nn.L1Loss(reduction="mean")

        self.teacher_model = LEDCurveNet()
        if teacher_weight:
            checkpoint = torch.load(teacher_weight)['params']
            self.teacher_model.load_state_dict(checkpoint)

    def forward(self, x, pred):
        with torch.no_grad():
            self.teacher_model.eval()
            gt_features, _ = self.teacher_model(x.detach())

        loss = self.l1_loss(pred, gt_features)

        return self.loss_weight * loss