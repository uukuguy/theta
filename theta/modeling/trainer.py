#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
from pathlib import Path
from loguru import logger

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from ..utils.progbar import Progbar
from ..utils.multiprocesses import is_master_process, is_multi_processes, barrier_leader_process, barrier_member_processes


def get_default_optimizer_parameters(model, weight_decay):
    param_optimizers = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weights']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizers
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizers if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_grouped_parameters


def generate_dataloader(args,
                        dataset,
                        batch_size,
                        keep_order=True,
                        collate_fn=None):

    Sampler = SequentialSampler if keep_order else RandomSampler
    sampler = DistributedSampler(dataset) if is_multi_processes(
        args) else Sampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    return dataloader


class Trainer:
    def __init__(self, args):
        self.args = args
        self.collate_fn = None

    #  def build_model(self, args):
    #      raise NotImplementedError

    def batch_to_inputs(self, args, batch, known_labels=True):
        raise NotImplementedError

    def examples_to_dataset(self, examples, max_seq_length):
        raise NotImplementedError

    #  def generate_dataloader(self, args, dataset, batch_size, keep_order=True):
    #
    #      return generate_dataloader(args, dataset, batch_size, keep_order)

    def on_train_step(self, args, model, step, batch):
        inputs = self.batch_to_inputs(args, batch)
        outputs = model(**inputs)
        return outputs

    def on_eval_start(self, args, eval_dataset):
        pass

    #  def on_eval_step(self, args, eval_dataset, step, model, inputs, outputs):
    #      results = {}
    #      return results

    def on_eval_step(self, args, model, step, batch, batch_features):
        inputs = self.batch_to_inputs(args, batch)
        outputs = model(**inputs)
        return outputs, {}

    def on_eval_end(self, args, eval_dataset):
        results = {}
        return results

    def on_predict_start(self, args, test_dataset):
        pass

    #  def on_predict_step(self, args, test_dataset, step, model, inputs,
    #                      outputs):
    def on_predict_step(self, args, model, step, batch):
        pass

    def on_predict_end(self, args, test_dataset):
        return []

    def save_model(self, args, model, tokenizer, optimizer, scheduler,
                   model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        logger.info(f"Saving model checkpoint to {model_path}")

        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(model_path)

        tokenizer.save_vocabulary(Path(model_path).as_posix() + '/')

        #  torch.save(args, os.path.join(model_path, "training_args.bin"))
        json.dump(
            {
                k: v
                for k, v in args.__dict__.items() if v is None
                or type(v) in [bool, str, int, float, dict, list, tuple]
            },
            open(os.path.join(model_path, "training_args.json"), 'w'),
            ensure_ascii=False, indent=2)

        torch.save(optimizer.state_dict(),
                   os.path.join(model_path, "optimizer.pt"))
        torch.save(scheduler.state_dict(),
                   os.path.join(model_path, "scheduler.pt"))

    def train(self, args, train_examples, eval_examples):

        logger.info(
            f"Start train: {len(train_examples)} train examples, {len(eval_examples)} eval examples."
        )

        #  if is_master_process(args):
        #      tb_writer = SummaryWriter()

        train_dataset, _ = self.examples_to_dataset(train_examples,
                                                    args.train_max_seq_length)
        train_dataloader = generate_dataloader(
            args,
            train_dataset,
            batch_size=args.per_gpu_train_batch_size,
            keep_order=False,
            collate_fn=self.collate_fn)
        logger.info(f"Start training ...")
        logger.info(f"  Num examples    = {len(train_examples)}")
        logger.info(f"  Num epoch steps = {len(train_dataloader)}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")
        logger.info(f"  Batch size = {args.per_gpu_train_batch_size}")

        steps_per_epoch = len(
            train_dataloader) // args.gradient_accumulation_steps

        if args.max_steps > 0:
            total_steps = args.max_steps
            args.num_train_epochs = args.max_steps // steps_per_epoch + 1
        else:
            total_steps = steps_per_epoch * args.num_train_epochs
        args.total_steps = total_steps

        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {total_steps}")

        model, optimizer, scheduler = self.build_model(args)
        tokenizer = self.tokenizer

        # Check if saved optimizer or scheduler states exist
        model_path = Path(args.model_path)
        optimizer_saved_file = model_path / "optimizer.pt"
        scheduler_saved_file = model_path / "scheduler.pt"
        if optimizer_saved_file.exists() and scheduler_saved_file.exists():
            optimizer.load_state_dict(torch.load(optimizer_saved_file))
            scheduler.load_state_dict(torch.load(scheduler_saved_file))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://github.com/nvidia/apex to use fp16."
                )
            model, optimizer = amp.initialize(model,
                                              optimizer,
                                              opt_level=args.fp16_opt_level)
            args.amp = amp
        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Distributed training (should be after apex fp16 initialization)
        #  if args.local_rank != -1:
        if is_multi_processes(args):
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True)

        trained_steps = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        model_path = Path(args.model_path)
        output_dir = Path(args.output_dir)

        if model_path.exists() and "checkpoint" in str(model_path):
            # set trained_steps to trained_steps of last saved checkpoint from model path
            trained_steps = int(model_path.parts()[-1].split("-")[-1])
            epochs_trained = trained_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = trained_steps % (
                len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d",
                        global_step)
            logger.info("  Will skip the first %d steps in the first epoch",
                        steps_trained_in_current_epoch)

        train_loss, logging_loss = 0.0, 0.0
        best_value = 0.0
        best_type = 'max'  # ['max', 'min']

        #  best_value = float('inf')
        #  best_type = 'min'

        model.zero_grad()

        #  train_iterator = trange(
        #      epochs_trained,
        #      int(args.num_train_epochs),
        #      desc="Epoch",
        #      disable=args.local_rank not in [-1, 0],
        #  )
        #  for epoch in train_iterator:
        for epoch in range(epochs_trained, args.num_train_epochs):

            #  epoch_iterator = tqdm(train_dataloader,
            #                        desc="Iteration",
            #                        disable=args.local_rank not in [-1, 0])
            #  for step, batch in enumerate(epoch_iterator):

            pbar = Progbar(target=len(train_dataloader),
                           stateful_metrics=['loss'],
                           desc=f"Epoch({epoch+1}/{args.num_train_epochs})")
            for step, batch in enumerate(train_dataloader):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)

                outputs = self.on_train_step(args, model, step, batch)
                #  inputs = self.batch_to_inputs(args, batch)
                #  outputs = model(**inputs)

                #  logger.debug(f"outputs: {outputs}")
                # -------- loss --------
                loss = outputs[0]
                #  logger.debug(f"loss: {loss}")
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                lr = scheduler.get_last_lr()[0]
                #  lr = scheduler.get_lr()[0]
                pbar.update(step + 1,
                            values=[('lr', lr), ('loss', loss.item())])
                train_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                trained_steps += 1

                # -------- Save models --------
                if is_master_process(
                        args) and trained_steps % steps_per_epoch == 0:

                    # -------- Save checkpoint --------
                    if args.save_checkpoints:
                        checkpoint_dir = f"checkpoint-{trained_steps}"
                        checkpoint_path = output_dir / checkpoint_dir
                        self.save_model(args, model, tokenizer, optimizer,
                                        scheduler, checkpoint_path)

                    # -------- Evaluate --------
                    if not args.no_eval_on_each_epoch:
                        logger.info(
                            f"Epoch({epoch+1}/{args.num_train_epochs}) evaluating."
                        )
                        eval_logs = {}
                        eval_results = self.evaluate(args, model,
                                                     eval_examples)
                        for key, value in eval_results.items():
                            eval_key = "eval_{}".format(key)
                            eval_logs[eval_key] = f"{value:.6f}"
                        loss_scalar = (train_loss -
                                       logging_loss) / steps_per_epoch
                        learning_rate_scalar = scheduler.get_last_lr()[0]
                        #  learning_rate_scalar = scheduler.get_lr()[0]
                        eval_logs[
                            "learning_rate"] = f"{learning_rate_scalar:.6f}"
                        eval_logs["loss"] = f"{loss_scalar:.6f}"
                        #  for key, value in eval_logs.items():
                        #      tb_writer.add_scalar(key, value, trained_steps)
                        print(
                            json.dumps({
                                **eval_logs,
                                **{
                                    "step": trained_steps
                                }
                            }))
                        logging_loss = train_loss

                        # -------- Save best model --------
                        best_index = 'f1'  # ['f1', 'acc', 'recall', 'loss']
                        eval_value = eval_results[best_index]
                        is_best = False
                        if best_index in ['loss']:
                            if eval_value < best_value:
                                is_best = True
                        else:
                            if eval_value > best_value:
                                is_best = True
                        if is_best:
                            logger.info(
                                f"Best {best_index}: {eval_value:.4f} ({eval_value - best_value:.6f})"
                            )
                            best_value = eval_value
                            bestmodel_path = output_dir / "best"
                            self.save_model(args, model, tokenizer, optimizer,
                                            scheduler, bestmodel_path)

                            if args.save_checkpoints:
                                best_symlink = output_dir / "best/best_checkpoint"
                                if best_symlink.is_symlink():
                                    best_symlink.unlink()
                                os.symlink(f"../{checkpoint_dir}",
                                           best_symlink)

            print(" ")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()

        #      if args.max_steps > 0 and trained_steps > args.max_steps:
        #          epoch_iterator.close()
        #          break
        #  if args.max_steps > 0 and trained_steps > args.max_steps:
        #      train_iterator.close()
        #      break

        #  if is_multi_processes(args):
        #      tb_writer.close()
        return trained_steps, train_loss / trained_steps

    def evaluate(self, args, model, eval_examples):
        eval_dataset, eval_features = self.examples_to_dataset(
            eval_examples, args.eval_max_seq_length)

        self.on_eval_start(args, eval_dataset)
        eval_dataloader = generate_dataloader(
            args,
            eval_dataset,
            batch_size=args.per_gpu_eval_batch_size,
            keep_order=True,
            collate_fn=self.collate_fn)

        eval_output_dir = args.output_dir
        if is_multi_processes(args) and not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        logger.info(f"Start evaluating ...")
        logger.info(f"  Num examples    = {len(eval_examples)}")
        logger.info(f"  Num epoch steps = {len(eval_dataloader)}")
        logger.info(f"  Batch size = {args.per_gpu_eval_batch_size}")

        eval_steps = 0
        eval_loss = 0.0
        pbar = Progbar(target=len(eval_dataloader),
                       stateful_metrics=['loss', 'acc', 'recall', 'f1'],
                       desc=f"Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                #  inputs = self.batch_to_inputs(args, batch)
                #  outputs = model(**inputs)
                batch_features = eval_features[args.per_gpu_eval_batch_size *
                                               step:args.
                                               per_gpu_eval_batch_size *
                                               (step + 1)]
                outputs, results = self.on_eval_step(args, model, step, batch,
                                                     batch_features)

                # -------- loss --------
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.item()

            eval_steps += 1

            #  results = self.on_eval_step(args, eval_dataset, step, model,
            #                              inputs, outputs)
            values = []
            if results:
                values = [(k, v) for k, v in results.items()]
            values += [('loss', loss.item())]
            pbar.update(step + 1, values=values)

        self.eval_loss = eval_loss / eval_steps
        results = self.on_eval_end(args, eval_dataset)

        return results

    def predict(self, args, model, test_examples):

        pred_output_dir = Path(args.pred_output_dir)
        if is_multi_processes(args) and not pred_output_dir.exists():
            os.makedirs(pred_output_dir)

        test_dataset, _ = self.examples_to_dataset(test_examples,
                                                   args.eval_max_seq_length)
        test_dataloader = generate_dataloader(
            args,
            test_dataset,
            batch_size=args.per_gpu_predict_batch_size,
            keep_order=True,
            collate_fn=self.collate_fn)

        logger.info(f"Start predicting ...")
        logger.info(f"  Num examples    = {len(test_examples)}")
        logger.info(f"  Num epoch steps = {len(test_dataloader)}")
        logger.info(f"  Batch size = {args.per_gpu_predict_batch_size}")

        pbar = Progbar(target=len(test_dataloader), desc=f"Predicting")
        self.on_predict_start(args, test_dataset)
        for step, batch in enumerate(test_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                #  inputs = self.package_inputs_from_batch(args,
                #                                          batch,
                #                                          known_labels=False)
                #  outputs = model(**inputs)
                #  self.on_predict_step(args, test_dataset, step, model, inputs,
                #                       outputs)
                self.on_predict_step(args, model, step, batch)

            pbar.update(step + 1)

        results = self.on_predict_end(args, test_dataset)
        return results
