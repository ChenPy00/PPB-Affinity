# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


from BaselineModel.modules.encoders.single import PerResidueEncoder
from BaselineModel.modules.encoders.pair import ResiduePairEncoder
from BaselineModel.modules.encoders.attn import GAEncoder
from BaselineModel.utils.protein.constants import BBHeavyAtom


class DG_Network(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        res_dim = cfg.encoder.node_feat_dim

        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=cfg.encoder.node_feat_dim,
            max_num_atoms=cfg.max_num_atoms,
        )

        self.single_fusion = nn.Sequential(
            nn.Linear(res_dim, res_dim), nn.ReLU(),
            nn.Linear(res_dim, res_dim)
        )
        self.mut_bias = nn.Embedding(
            num_embeddings=2,
            embedding_dim=res_dim,
            padding_idx=0,
        )
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=cfg.encoder.pair_feat_dim,
            max_num_atoms=cfg.max_num_atoms,
        )
        self.attn_encoder = GAEncoder(**cfg.encoder)

        # Pred
        self.predictor = nn.Sequential(
            nn.Linear(res_dim, res_dim), nn.ReLU(),
            nn.Linear(res_dim, res_dim), nn.ReLU(),
            nn.Linear(res_dim, cfg.num_classes)
        )

        self.num_classes = cfg.num_classes

    def encode(self, batch):
        N, L = batch['aa'].shape[:2]
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        x = self.single_encoder(
            aa=batch['aa'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue,
        )

        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b
        
        z = self.pair_encoder(
            aa=batch['aa'],
            res_nb=batch['res_nb'], chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'],
        )
        
        x = self.attn_encoder(
            pos_atoms=batch['pos_atoms'],
            res_feat=x, pair_feat=z,
            mask=mask_residue
        )

        return x

    def forward(self, batch):
        batch = {k: v for k, v in batch.items()}
        h = self.encode(batch)
        H = h.max(dim=1)[0]

        preds = self.predictor(H)
        labels_mask = batch['labels_mask']
        assert preds.size(1)==1 and batch['labels'].size(1)==1
        if preds.size()==batch['labels'].size():
            regression_loss = F.mse_loss(preds, batch['labels'], reduction='none')
            regression_loss = (regression_loss * labels_mask).sum() / labels_mask.sum()
        else:
            regression_loss = F.mse_loss(preds.unsqueeze(-1), batch['labels'], reduction='none')
            regression_loss = (regression_loss * labels_mask).sum() / labels_mask.sum()
        if regression_loss.isnan().any():
            import pdb;pdb.set_trace()

        loss_dict = {
            'regression': regression_loss
        }
        if self.num_classes > 1:
            out_dict = {
                'dG_pred': preds[:, 0],
                'dG_true': batch['dG']
            }
        else:
            out_dict = {
                'dG_pred': preds[:],
                'dG_true': batch['dG']
            }
        return loss_dict, out_dict

    def infer(self, batch):
        batch = {k: v for k, v in batch.items()}
        h = self.encode(batch)
        H = h.max(dim=1)[0]

        preds = self.predictor(H)
        return preds[:, 0] if self.num_classes > 1 else preds.squeeze(dim=-1)