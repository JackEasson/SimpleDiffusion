import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from models.egeunet import group_aggregation_bridge, LayerNorm
from models.ghostnetv2 import GhostModuleV2, SqueezeExcite, _make_divisible

"""
Source:
    https://github.com/HibikiJie/Diffusion-Models/blob/master/Model.py
"""
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class GhostBottleneckV2(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, tdim, dw_kernel_size=3,
                 stride=1, se_ratio=0., attention=False, args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if not attention:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='original', args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='attn', args=args)

            # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.temb_proj = nn.Sequential(
            nn.Linear(tdim, mid_chs),
            nn.GELU()
        )

        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False, mode='original', args=args)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x, temb):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x += self.temb_proj(temb)[:, :, None, None]
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, tdim, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

        self.temb_proj = nn.Sequential(
            nn.Linear(tdim, dim_out),
            nn.GELU()
        )

        self.ldw2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=dim_out),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, 1),
        )

    def forward(self, x, temb):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        # ----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # ----------dw----------#
        x4 = self.dw(x4)
        # ----------concat----------#
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        x += self.temb_proj(temb)[:, :, None, None]
        x = self.ldw2(x)
        return x


class GhostEGEUnet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 output_channels=3,
                 T=1000,
                 temb_channels=256,
                 c_list=[16, 24, 32, 48, 64, 96],
                 width=1.0,
                 final_activate=None,
                 bridge=True):
        super(GhostEGEUnet, self).__init__()

        self.bridge = bridge
        print('c_list', c_list, width)
        self.chn_list = [_make_divisible(chn * width, 4) for chn in c_list]
        self.temb_chn = temb_channels

        hidden_tdim = temb_channels // 2
        self.time_embedding = TimeEmbedding(T, hidden_tdim, temb_channels)

        self.head = nn.Sequential(
            nn.Conv2d(input_channels, self.chn_list[0], 3, stride=1, padding=1),
        )

        self.encoder_layers = self.get_encoder_layers(self.chn_list[0])
        # print(self.encoder_layers)

        self.decoder_layers = self.get_decoder_layers(self.chn_list[-1])

        self.gab_layers = self.get_gab_layers()

        self.final = nn.Sequential(
            nn.Conv2d(self.chn_list[0], self.chn_list[0], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, self.chn_list[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.GELU(),
            nn.Conv2d(self.chn_list[0], output_channels, kernel_size=3, stride=1, padding=1)
        )

        self.final_act = self.get_activate_layer(final_activate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def get_activate_layer(activate: str or None = None):
        assert activate is None or isinstance(activate, str)
        if activate is None:
            return nn.Identity()
        else:
            activate = activate.lower()
            if activate == 'sigmod':
                return nn.Sigmoid()
            elif activate == 'tanh':
                return nn.Tanh()
            else:
                raise ValueError(f'Error activate function for {activate}.')

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)

        x = self.head(x)

        # encoder modules
        encode_features = []
        for module in self.encoder_layers:
            for i, layer in enumerate(module):
                x = layer(x, temb) if i == 0 else layer(x)
            encode_features.append(x)
        # decoder modules
        last_gab_out = encode_features[-1]
        for i, module in enumerate(self.decoder_layers):
            for j, layer in enumerate(module):
                x = layer(x, temb) if j == 0 else layer(x)
            gab_out = self.gab_layers[-i - 1](last_gab_out, encode_features[-i - 2])
            x += gab_out
            last_gab_out = gab_out

        x = self.final_act(self.final(x))
        return x

    def get_encoder_layers(self, inp_chn):
        encoder_layers = nn.ModuleList()
        layers = len(self.chn_list)
        for i in range(layers):
            oup_chn = self.chn_list[i]
            encoder_module = nn.ModuleList()
            if i <= 2:
                mid_expand = 2
                encoder_module.append(GhostBottleneckV2(in_chs=inp_chn,
                                                        mid_chs=_make_divisible(inp_chn * mid_expand),
                                                        out_chs=oup_chn,
                                                        tdim=self.temb_chn,
                                                        dw_kernel_size=3,
                                                        stride=1,
                                                        attention=True))
            else:
                encoder_module.append(Grouped_multi_axis_Hadamard_Product_Attention(inp_chn, oup_chn, tdim=self.temb_chn))
            inp_chn = oup_chn
            if i != layers - 1:
                encoder_module.append(
                    nn.Sequential(
                        nn.GroupNorm(4, inp_chn),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        nn.GELU()
                    )
                )
            encoder_layers.append(encoder_module)
        return encoder_layers

    def get_decoder_layers(self, inp_chn):
        decoder_layers = nn.ModuleList()
        chn_list = self.chn_list[:-1][::-1]
        layers = len(chn_list)
        for i in range(layers):
            oup_chn = chn_list[i]
            decoder_module = nn.ModuleList()
            if i >= layers - 3:
                decoder_module.append(
                    Grouped_multi_axis_Hadamard_Product_Attention(inp_chn, oup_chn, tdim=self.temb_chn))
            else:
                mid_expand = 2
                decoder_module.append(GhostBottleneckV2(in_chs=inp_chn,
                                                        mid_chs=_make_divisible(inp_chn * mid_expand),
                                                        out_chs=oup_chn,
                                                        tdim=self.temb_chn,
                                                        dw_kernel_size=3,
                                                        stride=1,
                                                        attention=True))
            inp_chn = oup_chn
            if i != 0:
                decoder_module.append(
                    nn.Sequential(
                        nn.GroupNorm(4, inp_chn),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.GELU(),
                    )
                )
            decoder_layers.append(decoder_module)
        return decoder_layers

    def get_gab_layers(self):
        gab_layers = nn.ModuleList()
        for i in range(1, len(self.chn_list)):
            gab_layers.append(group_aggregation_bridge(self.chn_list[i], self.chn_list[i - 1]))
        return gab_layers


if __name__ == '__main__':
    unet = GhostEGEUnet(input_channels=3,
                        output_channels=3,
                        T=1000,
                        temb_channels=256,
                        c_list=[16, 24, 32, 48, 64, 96],
                        width=1.5,
                        final_activate=None,
                        bridge=True)
    unet.eval()
    torch.save(unet, 'egeunet.pth')
    with torch.no_grad():
        x = torch.randn((1, 3, 256, 256))
        t = torch.randint(1000, (1,))
        y = unet(x, t)
        print(y.size())
        print(y[0, 0, 0, :10])