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

logger = logging.get_logger(__name__)


class DeterministicLLMVAETrainer(Trainer):
    def __init__(
        self,
        normalize_output=False,
        e2e=False,
        train_gaussian=False,
        time_points=16,
        use_vae=False,
        straight_paths=False,
        target_dist=None,
        target_dim=5000,
        just_llm=False,
        train_2d=False,
        **args
    ):
        super().__init__(**args)
        self.time_points = time_points
        self.straight_paths = straight_paths

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
        X = torch.tensor(inputs['exprs']).to(model.device)
        if self.straight_paths:
            model_inputs = self.generate_straight_paths(X, device=model.device, time_points=self.time_points)
        else:
            model_inputs = self.generate_noise_paths(X, device=model.device, time_points=self.time_points)
        labels = model_inputs.detach().clone()
        token_masks = None
        


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, model_inputs, token_masks, labels)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, token_masks, labels, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        with autocast():
            outputs = model.cell_enc(inputs)
            outputs = model.gpt_neox(inputs_embeds=outputs)
            outputs = model.cell_dec(outputs.last_hidden_state)

            outputs_for_loss = outputs[:, :-1, :].contiguous()
            labels = labels[:, 1:, :].contiguous()

            loss_fn = MSELoss()
            loss = loss_fn(outputs_for_loss, labels)
            
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

        X = torch.tensor(inputs['exprs']).to(model.device)
        if self.straight_paths:
            model_inputs = self.generate_straight_paths(X, device=model.device, time_points=self.time_points)
        else:
            model_inputs = self.generate_noise_paths(X, device=model.device, time_points=self.time_points)
        labels = model_inputs.detach().clone()
        token_masks = None
        

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, model_inputs, token_masks, labels)
            loss = loss.mean().detach()

        return (loss, None, None)

    def generate_noise_paths(self, Y, sigma=0.0001, device='cuda', time_points=16):
        X = torch.normal(0.0, 1.0**0.5, size=Y.shape).to(device)
        W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)
        W[:, 0, :] = X
        W[:, 1, :] = Y
        # Define the time points, evenly spaced from 0 to 1, inclusive
        times = torch.linspace(0, 1, time_points)

        # Generate W where each slice W[:, i, :] is based on the linear combination of X and Y
        for i, t in enumerate(times[1:-1]):
            W[:, i, :] = torch.normal((1 - t) * X + t * Y, sigma**0.5).to(device)

        return W.to(dtype=torch.float32)

    def generate_straight_paths(self, Y, device='cuda', time_points=16):
        X = torch.normal(0.0, 1.0, size=Y.shape).to(device)
        W = torch.zeros(X.shape[0], time_points, X.shape[1]).to(device)

        # Define the time points, evenly spaced from 0 to 1, inclusive
        times = torch.linspace(0, 1, time_points)

        # Generate W where each slice W[:, i, :] is based on the linear combination of X and Y
        for i, t in enumerate(times):
            W[:, i, :] = (((1 - t) * X) + (t * Y)).to(device)

        return W.to(dtype=torch.float32)
