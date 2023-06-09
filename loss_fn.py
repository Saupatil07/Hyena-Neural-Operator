import torch
import torch.nn as nn
from einops import rearrange
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

def rel_loss(x, y, p, reduction=True, size_average=False, time_average=False):
    # x, y: [b, c, t, h, w] or [b, c, t, n]
    batch_num = x.shape[0]
    frame_num = x.shape[2]

    if len(x.shape) == 5:
        h = x.shape[3]
        w = x.shape[4]
        n = h*w
    else:
        n = x.shape[-1]
    # x = rearrange(x, 'b c t h w -> (b t h w) c')
    # y = rearrange(y, 'b c t h w -> (b t h w) c')
    num_examples = x.shape[0]
    eps = 1e-6
    diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, 1)
    y_norms = torch.norm(y.reshape(num_examples, -1), p, 1) + eps

    loss = torch.sum(diff_norms/y_norms)
    if reduction:
        loss = loss / batch_num
        if size_average:
            loss /= n
        if time_average:
            loss /= frame_num

    return loss


def rel_l2norm_loss(x, y):
    #   x, y [b, c, t, n]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-1) + eps
    diff = ((x-y)**2).mean(dim=-1)
    diff = diff / y_norm   # [b, c, t]
    diff = diff.sqrt().mean()
    return diff

def pointwise_rel_l2norm_loss(x, y):
    #   x, y [b, n, c]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-2) + eps
    diff = ((x-y)**2).mean(dim=-2)
    diff = diff / y_norm   # [b, c]
    diff = diff.sqrt().mean()
    return diff
