import json
import logging
import os

import torch
from torch import nn
from torch.distributions import Normal
from transformers import AutoTokenizer, AutoModel

from scvi.distributions import ZeroInflatedNegativeBinomial

class LLMVae(nn.Module):

    def __init__(self, lm, vae, n_hidden, freeze_vae=True):
        self.lm = lm
        self.vae = vae
        self.freeze_vae=freeze_vae
        if freeze_vae:
            for param in self.vae.module.parameters():
                param.requires_grad = False
        self.vae_proj = nn.Linear(n_hidden, lm.config.n_embd)
        self.lm_proj = nn.Linear(lm.config.n_embd, n_hidden)
        self.dispersion = vae.module.dispersion

    def forward(
        lm_token_ids,
        attention_mask,
        softmax_mask,
        expr_paths=None
        ):
        # Compute library sizes
        library = torch.log(X.sum(2)).unsqueeze(2)
        split_idx = lm_tokens_ids.shape[-1] # shape batch_size x num_tokens
        # LM embeddings
        lm_embs = model.transformer.wte(lm_token_ids)
        # VAE embeddings
        input_embs = lm_embs
        if expr_paths is not None:
            vae_embs = self.vae.z_encoder.encoder(expr_paths)
            vae_embs = self.vae_proj(vae_embs)

            input_embs = torch.cat((lm_embs, vae_embs), dim=1)
        outputs = self.lm(
            inputs_embeds=input_embs,
            attention_mask=attention_mask
            )
        output_embs = outputs.last_hidden_state
        
        text_embs = output_embs[:,:split_idx,:]
        expr_embs = output_embs[:,split_idx:,:]

        proj_expr_embs = self.lm_mean_proj(expr_embs)
        mean_embs = self.vae.z_encoder.mean_encoder(proj_expr_embs)
        var_embs = self.vae.z_encoder.var_activation(self.vae.z_encoder.var_encoder(proj_expr_embs)) + self.var_eps
        dist = Normal(mean_embs, var_embs.sqrt())
        vae_latents = self.vae.z_encoder.z_transformation(dist.rsample())
        cat_list = torch.zeros(vae_latents.shape[0], vae_latents.shape[1]).to('cuda')
        px_scale, px_r, px_rate, px_dropout = self.vae.decoder(
            self.dispersion,
            vae_latents,
            size_factor,
            cat_list
        )
        px_r = self.vae.px_r
        px_r = torch.exp(px_r)

        px = ZeroInflatedNegativeBinomial(
            mu=px_rate,
            theta=px_r,
            zi_logits=px_dropout,
            scale=px_scale,
        )

        samples = px.sample()

        text_logits = self.lm.lm_head(text_embs)
