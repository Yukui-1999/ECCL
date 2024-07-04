import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from model.vit_autoencoder import vit_autoencoder
from loss.clip_loss import ClipLoss
import argparse
class NC_method(nn.Module):
    def __init__(self, ecg_img_size=(12, 5000), ecg_patch_size=(1, 100),cmr_img_size=(80, 80), cmr_patch_size=(10, 10),
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, ecg_in_chans=1,cmr_in_chans=50,args=None):
        super(NC_method, self).__init__()
        self.ecg_encoder = vit_autoencoder(img_size=ecg_img_size, patch_size=ecg_patch_size,
                 embed_dim=embed_dim, depth=depth, num_heads=num_heads,latent_dim=args.latent_dim,
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, in_chans=ecg_in_chans)

        self.cmr_encoder = vit_autoencoder(img_size=cmr_img_size, patch_size=cmr_patch_size,
                    embed_dim=embed_dim, depth=depth, num_heads=num_heads,latent_dim=args.latent_dim,
                    decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, in_chans=cmr_in_chans)
        self.MSE_loss = nn.MSELoss()
        self.clip_loss = ClipLoss(temperature=args.temperature, alpha_weight=0.5,args=args)
        self.args = args
    def forward(self, ecg, cmr):
        rec_ecg, latent_ecg = self.ecg_encoder(ecg)
        rec_cmr, latent_cmr = self.cmr_encoder(cmr)

        rec_loss = self.MSE_loss(rec_ecg, ecg) + self.MSE_loss(rec_cmr, cmr)
        clip_loss = self.clip_loss(latent_ecg, latent_cmr)
        # print(f'rec_loss:{rec_loss},clip_loss:{clip_loss}')
        return rec_loss , clip_loss
# parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
# parser.add_argument('--temperature', default=0.1, type=float, help='Temperature parameter')
# parser.add_argument('--alpha_weight', default=0.5, type=float, help='Alpha weight')
# parser.add_argument('--lamda', default=1, type=float, help='Lambda parameter')
# parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# parser.add_argument('--device', default='cuda:0', type=str, help='Device')
#
# args = parser.parse_args()
# model = NC_method(ecg_img_size=(12, 5000), ecg_patch_size=(1, 100),cmr_img_size=(80, 80), cmr_patch_size=(10, 10),
#                     embed_dim=768, depth=12, num_heads=12,
#                     decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
#                     mlp_ratio=4., norm_layer=nn.LayerNorm, ecg_in_chans=1,cmr_in_chans=50,args=args)
# model = model.to(args.device)
# ecg = torch.randn(16, 1, 12, 5000).to(args.device)
# cmr = torch.randn(16, 50, 80, 80).to(args.device)
#
# output = model(ecg, cmr)
# print(output)