from model.MHatten import TransformerBlock
import torch.nn as nn
import torch


class SNPEncoder(nn.Module):
    def __init__(self, args):
        super(SNPEncoder, self).__init__()
        self.args = args
        self.norm = nn.LayerNorm(args.snp_size[1], eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.snp_size[1]))
        self.pos_embed = nn.Parameter(torch.zeros(1, args.snp_size[0] + 1, args.snp_size[1]))
        self.pos_drop = nn.Dropout(p=args.snp_drop_out)
        self.snp_attention = nn.Sequential(*[
            TransformerBlock(
                embed_size=args.snp_size[1],
                heads=8,
                dropout=args.snp_drop_out,
                forward_expansion=4
            ) for _ in range(args.snp_att_depth)
        ])
        self.projection = nn.Linear(args.snp_size[1], args.latent_dim)
    def forward(self, snp):
        B = snp.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print(f'cls_tokens.shape: {cls_tokens.shape}')
        # print(f'x.shape: {snp.shape}')
        x = torch.cat((cls_tokens, snp), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.snp_attention:
            x = blk(x,x,x)
        # 
        if self.args.snp_global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        x = self.projection(outcome)
        return x

