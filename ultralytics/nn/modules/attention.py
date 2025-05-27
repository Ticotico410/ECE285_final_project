import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import constant

__all__ = ["ADFFM_SpAtten", "ADFFM_ChAtten", 'ADFFM']


# from paddleseg.models import layers

class ConvBNReLU(nn.Module):
    """
    Conv + BN + ReLU
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int, optional): The kernel size of the conv layer. Default: 3.
        stride (int, optional): The stride of the conv layer. Default: 1.
        padding (int, optional): The padding of the conv layer. Default: 1.
        dilation (int, optional): The dilation of the conv layer. Default: 1.
        groups (int, optional): The groups of the conv layer. Default: 1.
        bias (bool, optional): Whether to use bias. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor.
        """
        conv_out = self.conv(x)
        bn_out = self.bn(conv_out)
        act_out = self.act(bn_out)

        return act_out


class ConvBNAct(nn.Module):
    """
    Conv + BN + Act
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int, optional): The kernel size of the conv layer. Default: 3.
        stride (int, optional): The stride of the conv layer. Default: 1.
        padding (int, optional): The padding of the conv layer. Default: 1.
        dilation (int, optional): The dilation of the conv layer. Default: 1.
        groups (int, optional): The groups of the conv layer. Default: 1.
        bias (bool, optional): Whether to use bias. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.LeakyReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor.
        """
        conv_out = self.conv(x)
        bn_out = self.bn(conv_out)
        act_out = self.act(bn_out)

        return act_out


class ConvBN(nn.Module):
    """
    Conv + BN + Act
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int, optional): The kernel size of the conv layer. Default: 3.
        stride (int, optional): The stride of the conv layer. Default: 1.
        padding (int, optional): The padding of the conv layer. Default: 1.
        dilation (int, optional): The dilation of the conv layer. Default: 1.
        groups (int, optional): The groups of the conv layer. Default: 1.
        bias (bool, optional): Whether to use bias. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor.
        """
        conv_out = self.conv(x)
        bn_out = self.bn(conv_out)

        return bn_out


class ADFFM(nn.Module):
    """
    The base of Attention-Driven Feature Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        # print(x_ch,y_ch,out_ch)
        self.conv_x = ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2)
        self.conv_out = ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1)

        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        # print(x_h, x_w, y_h, y_w)
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        # y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def fuse(self, x, y):
        out = x + y
        return out

    def forward(self, x):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x[1], x[0])
        x, y = self.prepare(x[1], x[0])
        out = self.fuse(x, y)
        # print(out.shape)
        return out


class ADFFM_ChAtten(ADFFM):
    """
    The ADFFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1, ),
            ConvBN(
                y_ch // 2, y_ch, kernel_size=1, ))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_hw([x, y], self.training)
        atten = torch.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ADFFM_SpAtten(ADFFM):
    """
    The ADFFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1),
            ConvBN(
                2, 1, kernel_size=3, padding=1))
        self._scale = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self._scale, 1.0) 

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        atten = torch.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out


def avg_reduce_hw(x):
    # Reduce hw by avg
    # Return cat([avg_pool_0, avg_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return F.adaptive_avg_pool2d(x, 1)
    elif len(x) == 1:
        return F.adaptive_avg_pool2d(x[0], 1)
    else:
        res = []
        for xi in x:
            res.append(F.adaptive_avg_pool2d(xi, 1))
        return torch.concat(res, dim=1)


def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when dim=[2, 3], the torch.max api has bug for training.
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = torch.max(x, dim=[2, 3], keepdim=True)

    if use_concat:
        res = torch.concat([avg_pool, max_pool], dim=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return torch.concat(res, dim=1)


def avg_reduce_channel(x):
    # Reduce channel by avg
    # Return cat([avg_ch_0, avg_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return torch.mean(x, dim=1, keepdim=True)
    elif len(x) == 1:
        return torch.mean(x[0], dim=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(torch.mean(xi, dim=1, keepdim=True))
        return torch.concat(res, dim=1)


def max_reduce_channel(x):
    # Reduce channel by max
    # Return cat([max_ch_0, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return torch.max(x, dim=1, keepdim=True)
    elif len(x) == 1:
        return torch.max(x[0], dim=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(torch.max(xi, dim=1, keepdim=True))
        return torch.concat(res, dim=1)


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True).values

    if use_concat:
        res = torch.concat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.concat(res, dim=1)


def cat_avg_max_reduce_channel(x):
    # Reduce hw by cat+avg+max
    assert isinstance(x, (list, tuple)) and len(x) > 1

    x = torch.concat(x, dim=1)

    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)
    res = torch.concat([mean_value, max_value], dim=1)

    return res
