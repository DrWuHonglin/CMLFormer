from re import S
from tkinter import SEL
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from torchvision.models import resnet18


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class MSC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.fc1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.act = nn.ReLU6()
        self.fc2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn_0 + attn_1 + attn_2

        # attn = self.conv3(attn) *u
        x = self.act(attn)
        x = self.fc2(x)

        return x


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class LMSA(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=3)

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        short = x
        # local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        # print("a",dots.shape)

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        # out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
        #       self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = attn + short
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class MLTB(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LMSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.jh = MSC(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.jh(self.norm2(x)))

        return x


class Spatial(nn.Module):
    def __init__(self, in_size, out_size):
        super(Spatial, self).__init__()
        self.conv1 = nn.Conv2d(out_size * 2, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        inputs2 = self.conv2(inputs2)
        outputs = torch.cat([inputs2, self.up(inputs1)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        # print("out",outputs.shape)
        return outputs


class Channel(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(Channel, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.conv = nn.Conv2d(decode_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.maxpool = nn.MaxPool2d(1, stride=1)
        # self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x + self.pre_conv(res))
        # x1,x2 = x.chunk(2,dim=1)
        x1 = self.maxpool(x)
        x2 = self.sigmoid(x)
        x = x1 + x2
        x = self.post_conv(x)
        # print("x",x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = MLTB(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = MLTB(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = Spatial(encoder_channels[-2], decode_channels)
        self.q3 = Channel(encoder_channels[-2], decode_channels)

        self.b2 = MLTB(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = Spatial(encoder_channels[-3], decode_channels)
        self.q2 = Channel(encoder_channels[-3], decode_channels)

        self.b1 = MLTB(dim=decode_channels, num_heads=8, window_size=window_size)
        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.aux_head = AuxHead(decode_channels, num_classes)
        self.p1 = Spatial(encoder_channels[-4], decode_channels)
        self.q1 = Channel(encoder_channels[-4], decode_channels)
        self.segmentation_head = nn.Sequential(ConvBNReLU(encoder_channels[1], decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        h4 = self.up4(x)
        # print("h",h4.shape)
        x = self.p3(x, res3) + self.q3(x, res3)
        x = self.b3(x)
        h3 = self.up3(x)
        # print("h",h3.shape)
        x = self.p2(x, res2) + self.q2(x, res2)
        x = self.b2(x)
        h2 = x
        # print("h",h2.shape)
        x = self.p1(x, res1) + self.q1(x, res1)
        x = self.b1(x)
        # print("x0",x.shape)
        ah = h4 + h3 + h2
        ah = self.up3(ah)
        # print("ah",ah.shape)
        x = torch.cat([x, ah], dim=1)
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        # print("ah",ah.shape)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.load_state_dict(torch.load('./pretrained/resnet18-118f1556.pth'))
        # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        x5 = self.model.avgpool(x4)

        # x = x.view(x.size(0), x.size(1))
        return x1, x2, x3, x4


class Unet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        self.resnet_features = base_resnet()

        encoder_channels = (64,128,256,512)

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.resnet_features(x)

        if self.training:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x


if __name__ == '__main__':
    x = torch.randn(6, 256, 256, 3)
    net = Unet(num_classes=6, pretrained=True)
    from thop import profile
    from thop import clever_format
    flops, params = profile(net, inputs=(x,))
    # print(flops, params) # 46388784.0 561706.0
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)  # 46.389M 561.706K
    # print(net)
    out = net(x)
    print(out.shape)