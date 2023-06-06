import logging
import os
import shutil
import sys
from typing import Any, Dict, List

import numpy as np
import torch

from .utils import create_batch


class End2EndBase:
    """This engine is for the end2end training.

    Args:
        working_dir (str): working directory.
    """

    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)

        self.train_modules = {}
        self.base_modules = {}
        self.batch_raw = {}
        self.global_steps = 0
        # Logging.
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S"
        )
        # std handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)
        # file handler
        file_handler = logging.FileHandler(os.path.join(working_dir, "running.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _build_scheduler(
        self, optimizer, step_size: int = 2, step_decay: float = 0.8
    ) -> torch.optim.lr_scheduler.StepLR:
        """schedule learning rate.

        Args:
            optimizer (_type_): optimizer
            step_size (int): Step size.
            step_decay (float): Step decay.

        Returns:
            torch.optim.lr_scheduler.StepLR: scheduler.
        """
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=step_decay,
        )
        return scheduler

    def _build_optimizer(self, params, learning_rate: float) -> torch.optim.Adam:
        """Build model optimizer.

        Args:
            params (_type_): model parameters.
            learning_rate (float): learning rate.

        Returns:
            torch.optim.Adam: optimizer.
        """
        return torch.optim.Adam(params, lr=learning_rate)

    def build_modules(
        self, base_modules: Dict[str, Any] = None, train_modules: Dict[str, Any] = None
    ):
        """Declare all modules in your model."""
        self.base_modules = {}
        self.train_modules = {}
        if base_modules:
            for model_name in base_modules:
                self.base_modules[model_name] = base_modules[model_name]
        if train_modules:
            for model_name in train_modules:
                self.train_modules[model_name] = train_modules[model_name]
        self._init_base_modules_params()

    def _init_base_modules_params(self):
        """Initialize Base Model Parameters."""
        for model_name in self.base_modules:
            file_model = f"{self.working_dir}/{model_name}.model"
            self.logger.info("Loading {}".format(file_model))
            try:
                self.base_modules[model_name].load_state_dict(
                    torch.load(file_model, map_location=lambda storage, loc: storage)
                )
            except Exception:
                self.logger.warning(f"Cannot Load {file_model}")

    def _build_pipeline(self):
        """Pipelines and loss here."""
        raise NotImplementedError

    def _build_batch(self, batch_raw: List[Any]):
        """process batch data."""
        raise NotImplementedError

    def _test_worker(self):
        """test worker"""
        raise NotImplementedError

    def _save_checkpoint(self, train_base_model: bool = True) -> None:
        """Save checkpoint models.

        Args:
            working_dir (str): Working Directory.
            epoch (int): current epoch.
            batch_id (int): current batch.
        """
        models_all = {}
        for key in self.base_modules:
            models_all[key] = self.base_modules[key]
        for key in self.train_modules:
            models_all[key] = self.train_modules[key]
        for model_name in models_all:
            fname = f"{self.working_dir}/{model_name}.model"
            self.logger.info(f"Saving {fname}.")
            fmodel = open(fname, "wb")
            torch.save(models_all[model_name].state_dict(), fmodel)
            fmodel.close()

    def fit(
        self,
        data_train: List[Dict[str, Any]],
        n_epoch: int = 10,
        batch_size: int = 10,
        learning_rate: float = 0.0001,
        lr_schedule: str = None,
        step_size: int = 2,
        step_decay: float = 0.8,
        warmup_step: int = 1000,
        train_base_model: str = False,
        grad_clip: float = 1.0,
        n_checkpoints: int = 3,
    ):
        """fit the model

        Args:
            data_train (List[Dict[str, Any]]): training data.
            n_epoch (int, optional): number of epoches. Defaults to 10.
            batch_size (int, optional): batch size. Defaults to 10.
            learning_rate (float, optional): learning rate. Defaults to 0.0001.
            lr_schedule (str, optional): learning rate schedule. Defaults to None.
            step_size (int, optional): step size in lr schedule. Defaults to 2.
            step_decay (float, optional): step decay in lr schedule. Defaults to 0.8.
            warmup_step (int, optional): warmup step. Defaults to 300.
            train_base_model (str, optional): train base models or not. Defaults to True.
            grad_clip (float, optional): gradient clip. Defaults to 0.5.
            n_checkpoints (int, optional): number of checkpoints in one epoch. Defaults to 3.
        """
        self.build_modules()
        self.logger.info(self.base_modules)
        self.logger.info(self.train_modules)
        # Parameters
        params = []
        for model_name in self.train_modules:
            params.extend(list(self.train_modules[model_name].parameters()))
        if train_base_model:
            for model_name in self.base_modules:
                params.extend(list(self.base_modules[model_name].parameters()))
        # Optimizer
        optimizer = self._build_optimizer(params, learning_rate)
        if lr_schedule == "decay":
            scheduler = self._build_scheduler(optimizer, step_size, step_decay)
        # begin to train.
        for epoch in range(n_epoch):
            batch_pool = create_batch(
                corpora=data_train,
                batch_size=batch_size,
                is_shuffle=True,
            )
            n_batches = len(batch_pool)
            self.logger.info("The number of batches: {}".format(n_batches))
            self.global_steps = n_batches * max(0, epoch)
            for batch_id, batch_raw in enumerate(batch_pool):
                self.global_steps += 1
                # learning rate schedule.
                if lr_schedule == "warm-up":
                    learning_rate = 2.0 * (
                        10000 ** (-0.5)
                        * min(
                            self.global_steps ** (-0.5),
                            self.global_steps * warmup_step ** (-1.5),
                        )
                    )
                    for p in optimizer.param_groups:
                        p["lr"] = learning_rate
                elif lr_schedule == "decay":
                    for p in optimizer.param_groups:
                        learning_rate = p["lr"]
                        break

                self._build_batch(batch_raw)
                loss = self._build_pipeline()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

                self.logger.info(
                    "epoch={}, batch={}, lr={}, loss={}".format(
                        epoch,
                        batch_id,
                        np.around(learning_rate, 6),
                        np.round(float(loss.data.cpu().numpy()), 4),
                    )
                )
                if batch_id == n_batches // n_checkpoints or batch_id == 0:
                    self._save_checkpoint(train_base_model)
                del loss
                torch.cuda.empty_cache()
            if epoch == n_epoch - 1:
                self._save_checkpoint(train_base_model)
            if lr_schedule == "decay":
                scheduler.step()
