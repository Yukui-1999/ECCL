import torch
import torch.nn as nn
import model.ECGEncoder as ECGEncoder
from pytorch_tabnet.pretraining import TabNetPretrainer
from model.tabnet_model import TabNetPretraining, create_group_matrix
import model.resnet as resnet
import torchvision
import copy
from model.swin_transformer import SwinTransformer
from loss.clip_loss import ClipLoss
from loss.triplet import TripletLoss
from itertools import combinations
import model.CMREncoder as CMREncoder
from model.mlpTablemodel import tabMlp
from model.SNPEncoder import SNPEncoder
class Trimodal_clip(nn.Module):
    def __init__(self, global_pool=False, device=None, tar_number=195, args=None) -> None:
        super().__init__()
        self.device = device
        self.args = args
        self.regression_linear = nn.Linear(args.Vit_embbeding, args.regression_dim)
        if 'ecg' in args.loss_type or args.loss_type == 'total':
            self.ECG_encoder = ECGEncoder.__dict__[args.ecg_model](
                img_size=args.ecg_input_size,
                patch_size=args.ecg_patch_size,
                in_chans=args.ecg_input_channels,
                num_classes=args.latent_dim,
                drop_rate=args.ecg_drop_out,
                args=args,
            )
            if args.ecg_pretrained:
                print("load pretrained ecg_model")
                ecg_checkpoint = torch.load(args.ecg_pretrained_model, map_location='cpu')
                ecg_checkpoint_model = ecg_checkpoint['model']
                msg = self.ECG_encoder.load_state_dict(ecg_checkpoint_model, strict=False)
                print('load ecg model')
                print(msg)
        if 'tar' in args.loss_type or args.loss_type == 'total':
        
            if args.tar_model == 'tabnet':
                self.TAR_encoder = TabNetPretraining(input_dim=tar_number,
                                                group_attention_matrix=create_group_matrix([], tar_number).to(device),
                                                device=device, latent_dim=args.latent_dim)
                if args.tar_pretrained:
                    print("load pretrained tar_model")
                    unsupervised_model_loaded = TabNetPretrainer()
                    unsupervised_model_loaded.load_model(args.tar_pretrained_path)
                    unsupervised_model_loaded.network.to(device)
                    print(f'unsupervised_model_loaded.device:{next(unsupervised_model_loaded.network.parameters()).device}')
                    self.TAR_encoder.load_state_dict(unsupervised_model_loaded.network.state_dict(),strict=False)
            elif args.tar_model == 'tabmlp':
                self.TAR_encoder = tabMlp(in_features=tar_number,hidden_features=args.tar_hidden_features,out_features=args.latent_dim,act_layer=nn.GELU,drop=args.tar_drop_out)

        if 'cmr' in args.loss_type or args.loss_type == 'total':
            if args.cmr_model.startswith('resnet'):
                self.CMR_encoder = resnet.__dict__[args.cmr_model](
                    in_channels=args.cmr_inchannels,
                    latent_dim=args.latent_dim,
                    pretrained=args.cmr_pretrained,
                )
            elif args.cmr_model.startswith('vit'):
                self.CMR_encoder = CMREncoder.__dict__[args.cmr_model](
                    in_chans=args.cmr_inchannels,
                    img_size=args.img_size,
                    num_classes=args.latent_dim,
                    drop_rate=args.cmr_drop_out,
                    args=args,
                )
                if args.cmr_pretrained:
                    print("load pretrained cmr_model")
                    cmr_checkpoint = torch.load(args.cmr_pretrained_model, map_location='cpu')
                    cmr_checkpoint_model = cmr_checkpoint['model']
                    self.CMR_encoder.load_state_dict(cmr_checkpoint_model, strict=False)
            elif args.cmr_model.startswith('swin'):
                self.CMR_encoder = SwinTransformer(img_size=args.img_size,
                        patch_size=(4, 4),
                        in_chans=args.cmr_inchannels,
                        num_classes=args.latent_dim,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.1,
                        attn_drop_rate=0.1,
                        drop_path_rate=0.2,
                        norm_layer=nn.LayerNorm,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False,
                        use_snp=args.use_snp,)
                if args.cmr_pretrained:
                    print("load pretrained cmr_model")
                    cmr_checkpoint = torch.load(args.cmr_pretrained_model, map_location='cpu')
                    cmr_checkpoint_model = cmr_checkpoint['model']
                    # 创建一个新的字典来存储不以'head'开头的参数
                    filtered_checkpoint_model = {k: v for k, v in cmr_checkpoint_model.items() if
                                                 not k.startswith('head')}
                    # 更新cmr_checkpoint_model
                    cmr_checkpoint_model = filtered_checkpoint_model
                    msg = self.CMR_encoder.load_state_dict(cmr_checkpoint_model, strict=False)
                    print(f'load cmr msg:{msg}')
        if 'snp' in args.loss_type or args.loss_type == 'total':
            self.SNP_encoder = SNPEncoder(args=args)

        if args.loss == 'triplet':
            self.loss_fn = TripletLoss(margin=args.margin)
        elif args.loss == 'clip_loss':
            self.loss_fn = ClipLoss(temperature=args.temperature, alpha_weight=args.alpha_weight, args=args)
        # def get_device(model):
        #     if hasattr(model, 'parameters'):
        #         try:
        #             next(iter(model.parameters()))
        #             return next(model.parameters()).device
        #         except StopIteration:
        #             return None
        #     else:
        #         return None
        # print(f'self.ECG_encoder device:{get_device(self.ECG_encoder)},self.TAR_encoder device:{get_device(self.TAR_encoder)},self.CMR_encoder:{get_device(self.CMR_encoder)},self.loss_fn:{get_device(self.loss_fn)}')
        # print(f'self.ECG_encoder device:{next(self.ECG_encoder.parameters()).device},self.TAR_encoder device:{next(self.TAR_encoder.parameters()).device},self.CMR_encoder:{next(self.CMR_encoder.parameters()).device},self.loss_fn:{next(self.loss_fn.parameters()).device}')

    def forward_loss(self, output_dict):
        all_combinations = combinations(output_dict.keys(), 2)
        loss_dict = {}

        for key_combination in all_combinations:
            # print(key_combination)
            loss_name = f"{key_combination[0][:-8]}_{key_combination[1][:-8]}_loss"
            loss_dict[loss_name] = self.loss_fn(output_dict[key_combination[0]], output_dict[key_combination[1]])
        loss_dict["total_loss"] = sum(loss_dict.values())
        
        return loss_dict["total_loss"]

    def forward(self, ecg, cmr, tar, snp):

        output_dict = {}
        if hasattr(self, 'ECG_encoder'):
            ecg_inter,ecg_feature = self.ECG_encoder(ecg)
            output_dict['ecg_feature'] = ecg_feature
        if hasattr(self, 'CMR_encoder'):
            cmr_inter,cmr_feature = self.CMR_encoder(cmr)
            output_dict['cmr_feature'] = cmr_feature
        if hasattr(self, 'TAR_encoder'):
            tar_feature = self.TAR_encoder(tar)
            output_dict['tar_feature'] = tar_feature
        if hasattr(self, 'SNP_encoder'):
            snp_feature = self.SNP_encoder(snp)
            output_dict['snp_feature'] = snp_feature

        # output_dict = {'ecg_feature': ecg_feature, 'cmr_feature': cmr_feature, 'tar_feature': tar_feature}
        
        loss = self.forward_loss(output_dict)
        ecg_regression = self.regression_linear(ecg_inter)
        
        return loss,ecg_regression
