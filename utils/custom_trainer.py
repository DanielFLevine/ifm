import os
import random
from typing import Dict, Optional, Union, Any, List, Tuple, NamedTuple
from collections.abc import Mapping

import datasets
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from packaging import version
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import is_datasets_available, _is_peft_model
from transformers.training_args import ParallelMode
from transformers.trainer_utils import seed_worker
from transformers.utils import (
    is_torch_tpu_available,
    logging,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from optimum.bettertransformer import BetterTransformer

from torch.distributions import Normal, kl_divergence
from scvi.distributions import ZeroInflatedNegativeBinomial

logger = logging.get_logger(__name__)

class _MODULE_KEYS(NamedTuple):
    X_KEY: str = "x"
    # inference
    Z_KEY: str = "z"
    QZ_KEY: str = "qz"
    QZM_KEY: str = "qzm"
    QZV_KEY: str = "qzv"
    LIBRARY_KEY: str = "library"
    QL_KEY: str = "ql"
    BATCH_INDEX_KEY: str = "batch_index"
    Y_KEY: str = "y"
    CONT_COVS_KEY: str = "cont_covs"
    CAT_COVS_KEY: str = "cat_covs"
    SIZE_FACTOR_KEY: str = "size_factor"
    # generative
    PX_KEY: str = "px"
    PL_KEY: str = "pl"
    PZ_KEY: str = "pz"
    # loss
    KL_L_KEY: str = "kl_divergence_l"
    KL_Z_KEY: str = "kl_divergence_z"


MODULE_KEYS = _MODULE_KEYS()


class LLMVAETrainer(Trainer):
    def __init__(
        self,
        max_context_length,
        max_num_blocks,
        normalize_output=False,
        e2e=False,
        train_gaussian=False,
        time_points=16,
        use_vae=False,
        straight_paths=False,
        target_dist=None,
        target_dim=5000,
        just_llm=False,
        train_custom=False,
        ifm_reg=False,
        kl_weight=1.0,
        sigma_decay=False,
        sigma_min=0.0001,
        sigma_max=0.01,
        dropout_p=0.0,
        scvi_dec=False,
        time_scale=False,
        alpha=2,
        min_weight=0.1,
        scale_last=False,
        scale_last_weight=10,
        **args
    ):
        super().__init__(**args)
        self.max_context_length = max_context_length
        self.max_num_blocks = max_num_blocks
        self.normalize_output = normalize_output
        self.e2e = e2e
        self.train_gaussian = train_gaussian
        self.time_points = time_points
        self.use_vae = use_vae
        self.straight_paths = straight_paths
        self.target_dist = target_dist
        self.target_dim = target_dim
        self.just_llm = just_llm
        self.train_custom = train_custom
        self.ifm_reg = ifm_reg
        self.kl_weight = kl_weight
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.dropout_p = dropout_p
        self.scvi_dec = scvi_dec
        self.time_scale = time_scale
        self.alpha = alpha
        self.min_weight = min_weight
        self.scale_last = scale_last
        self.scale_last_weight = scale_last_weight

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        # signature_columns = self._signature_columns
        signature_columns = ['prefix', 'path', 'expr']

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)


    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        # inputs = self._prepare_inputs(inputs)
        if self.train_gaussian:
            if self.target_dist in ('gaussian', 'checkerboard', 'bimodal'):
                batch_size = self.args.per_device_train_batch_size
                target_dim = self.target_dim
                if self.target_dist == 'checkerboard':
                    X = self.generate_checkerboard_samples(batch_size).to(model.device)
                elif self.target_dist == 'bimodal':
                    X = self.generate_bimodal_samples(batch_size).to(model.device)
                else:
                    X = torch.normal(0.0, 1.0, size=(batch_size, target_dim)).to(model.device)
            else:   
                X = torch.tensor(inputs['exprs']).to(model.device)
            if self.straight_paths:
                model_inputs = self.generate_straight_paths(X, device=model.device, time_points=self.time_points)
            else:
                model_inputs = self.generate_noise_paths(X, device=model.device, time_points=self.time_points, sigma=self.sigma_min)
            labels = model_inputs.detach().clone()
            token_masks = None
        else:
            input_embeds = []
            token_masks = []
            labels = []
            for i in range(len(inputs['prefixes'])):
                prefix = inputs['prefixes'][i]
                path = inputs['paths'][i]

                prefix_tens = torch.tensor(prefix, dtype=torch.int32).to(model.device)
                prefix_emb = model.gpt_neox.embed_in(prefix_tens)

                path_tens = torch.tensor(path).to(model.device)
                cat_input = torch.tensor([0]*path_tens.shape[0]).to(model.device)
                path_vae_emb = model.cell_enc(path_tens, cat_input)

                num_prefix_tokens = prefix_emb.shape[0]
                vae_hdim = path_vae_emb.shape[-1]

                path_emb = model.cell_proj_in(path_vae_emb)

                eos_emb = model.gpt_neox.embed_in(torch.tensor([0], dtype=torch.int32).to(model.device))


                token_mask = torch.tensor(
                    [0]*prefix_emb.shape[0] + [1]*path_emb.shape[0] + [0],
                    dtype=torch.int32
                    ).to(model.device)
                
                if self.e2e:
                    prefix_zeros = torch.zeros((num_prefix_tokens, path_tens.shape[-1]), requires_grad=False).to(model.device)
                    eos_zeros = torch.zeros((1, path_tens.shape[-1]), requires_grad=False).to(model.device)
                    label = torch.concat([prefix_zeros, path_tens.detach(), eos_zeros], axis=0)
                    labels.append(label)
                else:
                    prefix_zeros = torch.zeros((num_prefix_tokens, vae_hdim), requires_grad=False).to(model.device)
                    eos_zeros = torch.zeros((1, vae_hdim), requires_grad=False).to(model.device)
                    label = torch.concat([prefix_zeros, path_vae_emb.detach(), eos_zeros], axis=0)
                    labels.append(label)

                input_emb = torch.concat([prefix_emb, path_emb, eos_emb], axis=0)
                input_embeds.append(input_emb)
                token_masks.append(token_mask)

            input_embeds = torch.concat(input_embeds, axis=0)
            num_tokens = input_embeds.shape[0]
            num_blocks = min(num_tokens//self.max_context_length, self.max_num_blocks)
            input_embeds = input_embeds[:num_blocks*self.max_context_length].reshape(num_blocks, self.max_context_length, model.config.hidden_size)

            token_masks = torch.concat(token_masks, axis=0)
            token_masks = token_masks[:num_blocks*self.max_context_length].reshape(num_blocks, self.max_context_length)

            model_inputs = {'inputs_embeds': input_embeds}

            labels = torch.concat(labels, axis=0)
            labels = labels[:num_blocks*self.max_context_length].reshape(num_blocks, self.max_context_length, -1)


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, model_inputs, token_masks, labels, type='train')

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, token_masks, labels, type, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        with autocast():
            if self.train_gaussian:
                if self.just_llm:
                    outputs = model.gpt_neox(inputs_embeds=inputs).last_hidden_state
                    outputs = model.decoder(outputs)
                else:
                    outputs = model.cell_enc(inputs)
                    outputs = model.gpt_neox(inputs_embeds=outputs)
                    # if self.use_vae:
                    #     mean = model.mean_encoder(outputs.last_hidden_state)
                    #     var = model.var_encoder(outputs.last_hidden_state)
                    #     var = model.var_activation(var) + model.var_eps
                    #     cond_dist = Normal(mean, var.sqrt())
                    #     latent = cond_dist.sample()
                    #     outputs = model.decoder(latent)
                    if self.use_vae:
                        if self.scvi_dec:
                            px_scale, px_r, px_rate, px_dropout, latents, cond_dist = model.cell_dec(outputs.last_hidden_state)
                        else:
                            outputs, latents, cond_dist = model.cell_dec(outputs.last_hidden_state)
                    else:
                        outputs = model.cell_dec(outputs.last_hidden_state)
            else:
                outputs = model.gpt_neox(**inputs)
                outputs = model.cell_proj_out(outputs.last_hidden_state)
                if self.normalize_output:
                    outputs = model.output_norm(outputs)
                    outputs = model.output_relu(outputs)

            if (self.e2e) and (not self.train_gaussian):
                shifted_outputs = outputs[:, :-1, :]
                token_masks = token_masks[:, :-1]
                labels = labels[:, 1:, :]

                flat_labels = labels.reshape(-1, labels.shape[-1])
                flat_outputs = shifted_outputs.reshape(-1, shifted_outputs.shape[-1])
                flat_masks = token_masks.reshape(-1)
                flat_indices = torch.where(flat_masks)[0]
                selected_outputs = flat_outputs[flat_indices]
                x = flat_labels[flat_indices]
                library = torch.log(x.sum(1)).unsqueeze(1).to(model.device)
                inf_dict = self.vae_inference_dict(selected_outputs, model, library)
                gen_dict = self.generative_dict(inf_dict[MODULE_KEYS.Z_KEY], library, model)
                return self.vae_loss(x, inf_dict, gen_dict)
            else:
                if self.scvi_dec:
                    px = ZeroInflatedNegativeBinomial(
                        mu=px_rate[:,:-1,:],
                        theta=px_r,
                        zi_logits=px_dropout[:,:-1,:],
                        scale=px_scale[:,:-1,:]
                    )
                    labels = labels[:, 1:, :].contiguous()
                    loss = -px.log_prob(labels)
                else:
                    outputs_for_loss = outputs[:, :-1, :].contiguous()
                    if self.ifm_reg:
                        start = labels[:, 0, :]
                        end = labels[:, -1, :]
                        reg_weight = (end-start).unsqueeze(1)
                        reg_weight = reg_weight*0.1
                    labels = labels[:, 1:, :].contiguous()
                    if not self.train_gaussian:
                        token_masks = token_masks[:, :-1].contiguous().float()
                        outputs_for_loss = outputs_for_loss * token_masks.unsqueeze(-1)
                    reduction = 'mean'
                    if self.train_gaussian and self.use_vae:
                        reduction = 'none'
                    loss_fn = MSELoss(reduction=reduction)
                    if self.ifm_reg:
                        outputs_for_loss += outputs_for_loss*reg_weight
                    recon_loss = loss_fn(outputs_for_loss, labels)

                if self.train_gaussian and self.use_vae:
                    pz = Normal(torch.zeros_like(latents), torch.ones_like(latents))
                    kl_divergence_z = kl_divergence(
                        cond_dist,
                        pz
                    ).sum(dim=-1)[:,:-1]

                    recon_loss = recon_loss.sum(-1)

                    loss = recon_loss + (kl_divergence_z*self.kl_weight)

                    if self.time_scale:
                        loss_weight = (torch.linspace(0, 1, self.time_points-1)**self.alpha) + self.min_weight
                        loss = loss_weight.to(model.device) * loss
                    if self.scale_last:
                        loss[:,-1] *= self.scale_last_weight
                    loss = torch.mean(loss)
                    wandb.log(
                        data={
                            f"{type}/reconstruction loss": torch.mean(recon_loss),
                            f"{type}/kl loss": torch.mean(kl_divergence_z)
                        },
                        commit=False
                    )
                return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        if self.train_gaussian:
            if self.target_dist in ('gaussian', 'checkerboard', 'bimodal'):
                batch_size = self.args.per_device_eval_batch_size
                target_dim = self.target_dim
                if self.target_dist == 'checkerboard':
                    X = self.generate_checkerboard_samples(batch_size).to(model.device)
                elif self.target_dist == 'bimodal':
                    X = self.generate_bimodal_samples(batch_size).to(model.device)
                else:
                    X = torch.normal(0.0, 1.0, size=(batch_size, target_dim)).to(model.device)
            else:  
                X = torch.tensor(inputs['exprs']).to(model.device)
            if self.straight_paths:
                model_inputs = self.generate_straight_paths(X, device=model.device, time_points=self.time_points)
            else:
                model_inputs = self.generate_noise_paths(X, device=model.device, time_points=self.time_points, sigma=self.sigma_min)
            labels = model_inputs.detach().clone()
            token_masks = None
        else:
            input_embeds = []
            token_masks = []
            labels = []
            for i in range(len(inputs['prefixes'])):
                prefix = inputs['prefixes'][i]
                path = inputs['paths'][i]

                prefix_tens = torch.tensor(prefix, dtype=torch.int32).to(model.device)
                prefix_emb = model.gpt_neox.embed_in(prefix_tens)

                path_tens = torch.tensor(path).to(model.device)
                cat_input = torch.tensor([0]*path_tens.shape[0]).to(model.device)
                path_vae_emb = model.cell_enc(path_tens, cat_input)

                num_prefix_tokens = prefix_emb.shape[0]
                vae_hdim = path_vae_emb.shape[-1]
                prefix_zeros = torch.zeros((num_prefix_tokens, vae_hdim), requires_grad=False).to(model.device)

                path_emb = model.cell_proj_in(path_vae_emb)

                eos_emb = model.gpt_neox.embed_in(torch.tensor([0], dtype=torch.int32).to(model.device))
                eos_zeros = torch.zeros((1, vae_hdim), requires_grad=False).to(model.device)


                token_mask = torch.tensor(
                    [0]*prefix_emb.shape[0] + [1]*path_emb.shape[0] + [0],
                    dtype=torch.int32
                    ).to(model.device)

                if self.e2e:
                    prefix_zeros = torch.zeros((num_prefix_tokens, path_tens.shape[-1]), requires_grad=False).to(model.device)
                    eos_zeros = torch.zeros((1, path_tens.shape[-1]), requires_grad=False).to(model.device)
                    label = torch.concat([prefix_zeros, path_tens.detach(), eos_zeros], axis=0)
                    labels.append(label)
                else:
                    prefix_zeros = torch.zeros((num_prefix_tokens, vae_hdim), requires_grad=False).to(model.device)
                    eos_zeros = torch.zeros((1, vae_hdim), requires_grad=False).to(model.device)
                    label = torch.concat([prefix_zeros, path_vae_emb.detach(), eos_zeros], axis=0)
                    labels.append(label)

                input_emb = torch.concat([prefix_emb, path_emb, eos_emb], axis=0)
                input_embeds.append(input_emb)
                token_masks.append(token_mask)

            input_embeds = torch.concat(input_embeds, axis=0)
            num_tokens = input_embeds.shape[0]
            num_blocks = min(num_tokens//self.max_context_length, self.max_num_blocks)
            input_embeds = input_embeds[:num_blocks*self.max_context_length].reshape(num_blocks, self.max_context_length, model.config.hidden_size)

            token_masks = torch.concat(token_masks, axis=0)
            token_masks = token_masks[:num_blocks*self.max_context_length].reshape(num_blocks, self.max_context_length)

            model_inputs = {'inputs_embeds': input_embeds}
            labels = torch.concat(labels, axis=0)
            labels = labels[:num_blocks*self.max_context_length].reshape(num_blocks, self.max_context_length, -1)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, model_inputs, token_masks, labels, type='eval')
            loss = loss.mean().detach()

        return (loss, None, None)

    def vae_inference_dict(self, q, model, library):

        q_m = model.mean_encoder(q)
        q_v = model.var_activation(model.var_encoder(q)) + model.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = model.z_transformation(dist.rsample())
        ql = None

        return {
            MODULE_KEYS.Z_KEY: latent,
            MODULE_KEYS.QZ_KEY: dist,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
        }

    def generative_dict(self, z, library, model):

        cat_list = torch.zeros(z.shape[0], z.shape[1]).to(model.device)

        px_scale, px_r, px_rate, px_dropout = model.decoder(
            'gene',
            z,
            library,
            cat_list
        )

        px_r = model.px_r

        px_r = torch.exp(px_r)
        px = ZeroInflatedNegativeBinomial(
            mu=px_rate,
            theta=px_r,
            zi_logits=px_dropout,
            scale=px_scale,
        )

        pl = None
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }

    def vae_loss(self, x, inference_outputs, generative_outputs, kl_weight=1.0):
        kl_divergence_z = kl_divergence(
            inference_outputs[MODULE_KEYS.QZ_KEY],
            generative_outputs[MODULE_KEYS.PZ_KEY]
        ).sum(dim=-1)

        reconst_loss = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x).sum(-1)
        weighted_kl_local = kl_weight * kl_divergence_z
        loss = torch.mean(reconst_loss + weighted_kl_local)

        return loss

    def generate_noise_paths(self, Y, sigma=0.0001, device='cuda', time_points=16):
        X = torch.normal(0.0, 1.0**0.5, size=Y.shape).to(device)
        W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)
        W[:, 0, :] = X
        W[:, 1, :] = Y
        # Define the time points, evenly spaced from 0 to 1, inclusive
        times = torch.linspace(0, 1, time_points).to(device)

        # Generate W where each slice W[:, i, :] is based on the linear combination of X and Y
        for i, t in enumerate(times[1:-1]):
            if self.sigma_decay:
                sigma = ((1-t)*self.sigma_max) + (t*self.sigma_min)
            W[:, i, :] = torch.normal((1 - t) * X + t * Y, sigma**0.5).to(device)

        return W.to(dtype=torch.float32)

    def generate_straight_paths(self, Y, device='cuda', time_points=16):
        X = torch.normal(0.0, 1.0, size=Y.shape).to(device)
        if self.scvi_dec:
            X = F.dropout(X, p=0.8, training=True)
        W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)

        # Define the time points, evenly spaced from 0 to 1, inclusive
        times = torch.linspace(0, 1, time_points)

        # Generate W where each slice W[:, i, :] is based on the linear combination of X and Y
        for i, t in enumerate(times):
            W[:, i, :] = (((1 - t) * X) + (t * Y)).to(device)
        
        W = W.to(dtype=torch.float32)
        if self.dropout_p > 0.0:
            W = F.dropout(W, p=self.dropout_p, training=True)
        return W

    def generate_checkerboard_samples(self, num_samples):
        # Initialize array to store samples
        samples = np.zeros((num_samples, 2))
        
        # List of supported squares, each defined by the lower bound corner (x, y)
        supported_squares = []
        for x in range(-4, 4, 2):  # from -4 to 2, stepping by 2
            for y in range(-4, 4, 2):  # from -4 to 2, stepping by 2
                if (x // 2 + y // 2) % 2 == 0:  # checkerboard pattern condition
                    supported_squares.append((x, y))
        
        # Generate samples
        for i in range(num_samples):
            # Choose a random square from the supported squares
            square = supported_squares[np.random.randint(len(supported_squares))]
            # Generate a random point within this square
            samples[i, 0] = np.random.uniform(square[0], square[0] + 2)
            samples[i, 1] = np.random.uniform(square[1], square[1] + 2)
        
        return torch.tensor(samples)

    def generate_bimodal_samples(self, num_samples):
        # Define the means for the two modes
        num_samples_mode = num_samples // 2
        mean1 = torch.tensor([[-2.0, -2.0]]*num_samples_mode)
        mean2 = torch.tensor([[2.0, 2.0]]*num_samples_mode)

        # Standard deviation for both modes (standard normal)
        std = torch.tensor([[1.0, 1.0]]*num_samples_mode)

        # Half samples from each mode
        num_samples_mode = num_samples // 2

        # Generate samples for each mode
        samples_mode1 = torch.normal(mean1, std)
        samples_mode2 = torch.normal(mean2, std)

        # Concatenate samples from both modes
        samples = torch.cat((samples_mode1, samples_mode2), dim=0)

        # Shuffle the samples to mix modes
        indices = torch.randperm(num_samples)
        shuffled_samples = samples[indices]

        return shuffled_samples
