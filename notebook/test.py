from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.unet_2d_blocks import UNetMidBlock2DCrossAttn


import torch
import torch.nn as nn
from diffusers.models.resnet import Downsample2D, FirDownsample2D, FirUpsample2D, KDownsample2D, KUpsample2D, ResnetBlock2D, Upsample2D

from diffusers.models.attention_processor import Attention
num_attention_heads = 4
attention_head_dim = 16
inner_dim = num_attention_heads * attention_head_dim

feature = torch.randn(3, 64, 8, 8)
fr_emb = torch.randn(3, 64, 64)

a = Attention(64, cross_attention_dim=64, dim_head=32)



class UNetMidBlock2DCrossAttnImageKV(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        add_attention: bool = True,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads


        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]

        cross_attentions = []

        for _ in range(num_layers):
            cross_attentions.append(Transformer2DModel(
                num_attention_heads = num_attention_heads,
                attention_head_dim = in_channels // num_attention_heads,
                in_channels = in_channels,
                num_layers=1,
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                only_cross_attention = False,
                # use_linear_projection=use_linear_projection,
                # upcast_attention=upcast_attention,
            ))

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.cross_attentions = nn.ModuleList(cross_attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        
        hidden_states = self.resnets[0](hidden_states, temb)
        print(hidden_states.shape)
        for attn, resnet in zip(self.cross_attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states,
                                     encoder_hidden_states = encoder_hidden_states,
                                     return_dict=False
                                     )[0]
                hidden_states = resnet(hidden_states, temb)

        return hidden_states


mid = UNetMidBlock2DCrossAttnImageKV(in_channels=64,
                               temb_channels=512,
                               cross_attention_dim=128 
                               )
mid


temb=torch.randn(3, 512)
feature = torch.randn(3, 64, 32, 32)
fr_emb = torch.randn(3, 32*32, 128)

out = mid(hidden_states=feature,
    encoder_hidden_states=fr_emb,
    temb=temb)

# 전달해야하는 형태는 [B, seq, dim]

print(out.shape)


from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D



down = CrossAttnDownBlock2D(in_channels=32,
                            out_channels=128,
                            temb_channels=512,
                            transformer_layers_per_block=1,
                            cross_attention_dim=768,
                            only_cross_attention=False,
                            )

down2 = CrossAttnDownBlock2D(in_channels=128,
                            out_channels=256,
                            temb_channels=512,
                            transformer_layers_per_block=1,
                            cross_attention_dim=768,
                            only_cross_attention=False,
                            )

temb=torch.randn(3, 512)
feature = torch.randn(3, 32, 128, 128)
fr_emb = torch.randn(3, 32*32, 768)


down_block_res_samples = (feature,)

sample, res_samples = down(hidden_states = feature, encoder_hidden_states=fr_emb, temb=temb)
print(sample.shape)
print(len(res_samples))
print(res_samples[0].shape, res_samples[1].shape)
down_block_res_samples += res_samples


sample, res_samples = down2(hidden_states = sample, encoder_hidden_states=fr_emb, temb=temb)
print(sample.shape)
print(len(res_samples))
print(res_samples[0].shape, res_samples[1].shape)
down_block_res_samples += res_samples