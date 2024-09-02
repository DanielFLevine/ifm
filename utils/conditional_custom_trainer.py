import json
from typing import Dict, Optional, Union, Any, List, Tuple
from collections.abc import Mapping

import datasets
import torch
import wandb
from packaging import version
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers.utils import (
    logging,
)
from ipdb import set_trace

from torch.distributions import Normal, kl_divergence

logger = logging.get_logger(__name__)


class ConditionalLLMVAETrainer(Trainer):
    def __init__(
        self,
        max_context_length,
        max_num_blocks,
        time_points=16,
        straight_paths=False,
        kl_weight=1.0,
        sigma_decay=False,
        sigma_min=0.0001,
        sigma_max=0.01,
        min_weight=0.1,
        space_dim=1,
        points_per_sample=1,
        prompt_path=None,
        tokenizer=None,
        samples_per_input=1,
        **args
    ):
        super().__init__(**args)
        self.max_context_length = max_context_length
        self.max_num_blocks = max_num_blocks
        self.time_points = time_points
        self.straight_paths = straight_paths
        self.kl_weight = kl_weight
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.min_weight = min_weight
        self.space_dim = space_dim
        self.points_per_sample = points_per_sample

        self.prompt_path = prompt_path
        if self.prompt_path:
            with open(self.prompt_path, "r") as f:
                self.prompts = json.load(f)
        else:
            self.prompts = None

        self.tokenizer = tokenizer
        self.samples_per_input = samples_per_input

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
        signature_columns = ['expression', 'cell_type', 'perturbation', 'chronicity']

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
        prefix_tokens = self.build_and_tokenize_prefixes(inputs, self.prompts, self.tokenizer)
        prefix_input_ids = prefix_tokens['prefix_input_ids']
        prefix_tokens_attention_mask = prefix_tokens['prefix_attention_mask'] # shape batch_size x seq_len
        self.prefix_length = prefix_input_ids.shape[1]
        path_tens = self.generate_straight_paths(inputs['expression'], device=model.device, time_points=self.time_points)
        path_tens = self.multipoint_reshape(path_tens, self.points_per_sample)

        labels = path_tens.detach().clone()

        with torch.no_grad():
            prefix_emb = model.gpt_neox.embed_in(prefix_input_ids) # shape batch_size x seq_len x hidden_dim

        path_emb = model.cell_enc(path_tens) # shape batch_size x time_points*points_per_sample x hidden_dim
        # Reshape for spatial integration
        batch_size, seq_len, feature = path_emb.shape
        path_emb = path_emb.view(batch_size, seq_len, self.space_dim, feature // self.space_dim)
        path_emb = path_emb.view(batch_size, seq_len* self.space_dim, feature // self.space_dim)
        input_emb = torch.concat([prefix_emb, path_emb], dim=1)
        
        # Extend prefix_tokens_attention_mask to match the length of input_emb
        attention_mask = torch.cat(
            [
                prefix_tokens_attention_mask,
                torch.ones((path_emb.shape[0], path_emb.shape[1]), dtype=torch.int32).to(model.device)
            ], 
            dim=1
        )

        # Reshape so that multiple samples are in each row
        batch_size = input_emb.shape[0]
        assert batch_size % self.samples_per_input == 0, "Batch size must be divisible by the number of samples per input"

        new_batch_size = batch_size // self.samples_per_input

        input_emb = input_emb.view(new_batch_size, self.samples_per_input, -1, input_emb.shape[-1])
        input_emb = input_emb.permute(0, 2, 1, 3).reshape(new_batch_size, -1, input_emb.shape[-1])

        attention_mask = attention_mask.view(new_batch_size, self.samples_per_input, -1)
        attention_mask = attention_mask.permute(0, 2, 1).reshape(new_batch_size, -1)

        labels = labels.view(new_batch_size, self.samples_per_input, -1, labels.shape[-1])
        labels = labels.permute(0, 2, 1, 3).reshape(new_batch_size, -1, labels.shape[-1])


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, input_emb, attention_mask, labels, type='train')

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, attention_mask, labels, type, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        with autocast():

            outputs = model.gpt_neox(attention_mask=attention_mask, inputs_embeds=inputs).last_hidden_state # [batch_size, prefix_leng+num_time_points*space_dim, d_model]
            outputs = outputs[:, self.prefix_length:, ...]

            outputs, latents, cond_dist = model.cell_dec(outputs) # [batch_size, n_timepoint, feature_dim]
            
            # last points_per_sample correspond to final time point, so we ignore those
            outputs_for_loss = outputs[:, -labels.shape[1]:-self.points_per_sample, :].contiguous()
            labels = labels[:, self.points_per_sample:, :].contiguous()
            reduction = 'none'
            loss_fn = MSELoss(reduction=reduction)
            recon_loss = loss_fn(outputs_for_loss, labels)

            pz = Normal(torch.zeros_like(latents), torch.ones_like(latents))
            kl_divergence_z = kl_divergence(
                cond_dist,
                pz
            ).sum(dim=-1)[:,:-self.space_dim*self.points_per_sample]

            recon_loss = torch.mean(recon_loss.sum(-1))
            kl_loss = torch.mean(kl_divergence_z)
            loss = recon_loss + (kl_loss*self.kl_weight)


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
        with torch.no_grad():
            with autocast():
                prefix_tokens = self.build_and_tokenize_prefixes(inputs, self.prompts, self.tokenizer)
                prefix_input_ids = prefix_tokens['prefix_input_ids']
                prefix_tokens_attention_mask = prefix_tokens['prefix_attention_mask'] # shape batch_size x seq_len

                path_tens = self.generate_straight_paths(inputs['expression'], device=model.device, time_points=self.time_points)
                path_tens = self.multipoint_reshape(path_tens, self.points_per_sample)

                labels = path_tens.detach().clone()

                prefix_emb = model.gpt_neox.embed_in(prefix_input_ids) # shape batch_size x seq_len x hidden_dim

                path_emb = model.cell_enc(path_tens) # shape batch_size x time_points*points_per_sample x hidden_dim
                # Reshape for spatial integration
                batch_size, seq_len, feature = path_emb.shape
                path_emb = path_emb.view(batch_size, seq_len, self.space_dim, feature // self.space_dim)
                path_emb = path_emb.view(batch_size, seq_len* self.space_dim, feature // self.space_dim)
                input_emb = torch.concat([prefix_emb, path_emb], dim=1)

                # Extend prefix_tokens_attention_mask to match the length of input_emb
                attention_mask = torch.cat(
                    [
                        prefix_tokens_attention_mask,
                        torch.ones((path_emb.shape[0], path_emb.shape[1]), dtype=torch.int32).to(model.device)
                    ], 
                    dim=1
                )

                # Reshape so that multiple samples are in each row
                batch_size = input_emb.shape[0]
                assert batch_size % self.samples_per_input == 0, "Batch size must be divisible by the number of samples per input"

                new_batch_size = batch_size // self.samples_per_input

                input_emb = input_emb.view(new_batch_size, self.samples_per_input, -1, input_emb.shape[-1])
                input_emb = input_emb.permute(0, 2, 1, 3).reshape(new_batch_size, -1, input_emb.shape[-1])

                attention_mask = attention_mask.view(new_batch_size, self.samples_per_input, -1)
                attention_mask = attention_mask.permute(0, 2, 1).reshape(new_batch_size, -1)

                labels = labels.view(new_batch_size, self.samples_per_input, -1, labels.shape[-1])
                labels = labels.permute(0, 2, 1, 3).reshape(new_batch_size, -1, labels.shape[-1])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, input_emb, attention_mask, labels, type='eval')
            loss = loss.mean().detach()

        return (loss, None, None)
    
    def build_and_tokenize_prefixes(self, inputs, prompts, tokenizer, max_length=1024, device='cuda'):
        batch_size = len(inputs['perturbation'])
        prefixes = []
        for i in range(batch_size):
            cell_type = inputs['cell_type'][i]
            perturbation = inputs['perturbation'][i]
            chronicity = inputs['chronicity'][i]
            if perturbation == 'No stimulation':
                prefix = prompts['prefix']['control'][0].format(cell_type=cell_type, chronicity=chronicity)
            else:
                prefix = prompts['prefix']['perturbation'][0].format(cell_type=cell_type, perturbation=perturbation, chronicity=chronicity)
            prefixes.append(prefix)
        
        tokenized = tokenizer(
            prefixes, 
            truncation=True, 
            max_length=max_length, 
            padding='longest', 
            return_tensors='pt',
        )
        
        return {
            'prefix_input_ids': tokenized['input_ids'].to(device),
            'prefix_attention_mask': tokenized['attention_mask'].to(device)
        }

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
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, device=device)
        X = torch.normal(0.0, 1.0, size=Y.shape).to(device)
        W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)

        # Define the time points, evenly spaced from 0 to 1, inclusive
        times = torch.linspace(0, 1, time_points)

        # Generate W where each slice W[:, i, :] is based on the linear combination of X and Y
        for i, t in enumerate(times):
            W[:, i, :] = (((1 - t) * X) + (t * Y)).to(device)
        
        W = W.to(dtype=torch.float32)
        return W

    def multipoint_reshape(self, X, points_per_sample):

        batch_size, seq_len, feature_dim = X.shape

        new_batch_size = batch_size//points_per_sample

        # Reshape and permute the tensor
        x_reshaped = X.view(new_batch_size, points_per_sample, seq_len, feature_dim).permute(0, 2, 1, 3).reshape(new_batch_size, points_per_sample*seq_len, feature_dim)
        return x_reshaped

