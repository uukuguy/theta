#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
from loguru import logger
#  from rich import print


class NerApp:
    def __init__(self,
                 experiment_params,
                 ner_labels,
                 ner_connections,
                 add_special_args=None):
        self.experiment_params = experiment_params
        self.ner_labels = ner_labels
        self.ner_connections = ner_connections

        if experiment_params.ner_params.ner_type == 'span':
            from .ner_span import load_model, get_args, NerTrainer
        elif experiment_params.ner_params.ner_type == 'pn':
            from .ner_pn import load_model, get_args, NerTrainer
        else:
            from .ner import load_model, get_args, NerTrainer

        args = get_args(experiment_params=experiment_params,
                        special_args=[add_special_args])
        self.args = args
        logger.info(f"args: {args}")
        #  print("[bold cyan]args:[/bold cyan]", args)

        self.trainer = None

    def get_trainer(self):
        # -------------------- Model --------------------
        if self.trainer is None:
            args = self.args
            if args.ner_type == 'span':
                from .ner_span import NerTrainer
            elif args.ner_type == 'pn':
                from .ner_pn import NerTrainer
            else:
                from .ner import NerTrainer

            class AppTrainer(NerTrainer):
                def __init__(self, args, ner_labels):
                    super(AppTrainer, self).__init__(args,
                                                     ner_labels,
                                                     build_model=None)

                #  def on_predict_end(self, args, test_dataset):
                #      super(Trainer, self).on_predict_end(args, test_dataset)

            self.trainer = AppTrainer(args, self.ner_labels)

        return self.trainer

    def load_model(self):
        if self.experiment_params.ner_params.ner_type == 'span':
            from .ner_span import load_model as global_load_model
        elif self.experiment_params.ner_params.ner_type == 'pn':
            from .ner_pn import load_model as global_load_model
        else:
            from .ner import load_model as global_load_model

        model = global_load_model(self.args)
        return model

    def run(self,
            train_data_generator,
            test_data_generator,
            generate_submission=None,
            eval_data_generator=None):
        args = self.args

        assert train_data_generator is not None
        assert test_data_generator is not None
        if eval_data_generator is None:
            eval_data_generator = train_data_generator

        def do_eda(args):
            from .ner_utils import show_ner_datainfo
            show_ner_datainfo(self.ner_labels, train_data_generator,
                              args.train_file, test_data_generator,
                              args.test_file)

        def do_submit(args):
            if generate_submission:
                submission_file = generate_submission(args)
                from .utils import archive_local_model
                archive_local_model(args, submission_file)

        if args.do_eda:
            do_eda(args)

        elif args.do_submit:
            do_submit(args)

        elif args.to_train_poplar:
            from .ner_utils import to_train_poplar
            to_train_poplar(args,
                            train_data_generator,
                            ner_labels=ner_labels,
                            ner_connections=ner_connections,
                            start_page=args.start_page,
                            max_pages=args.max_pages)

        elif args.to_reviews_poplar:
            from .ner_utils import to_reviews_poplar
            to_reviews_poplar(args,
                              ner_labels=ner_labels,
                              ner_connections=ner_connections,
                              start_page=args.start_page,
                              max_pages=args.max_pages)

        else:

            trainer = self.get_trainer()

            from .ner_utils import load_train_val_examples, load_test_examples

            def do_train(args):
                train_examples, val_examples = load_train_val_examples(
                    args, train_data_generator, self.ner_labels)
                trainer.train(args, train_examples, val_examples)

            def do_eval(args):
                args.model_path = args.best_model_path
                _, eval_examples = load_train_val_examples(
                    args, train_data_generator, self.ner_labels)
                model = self.load_model()
                trainer.evaluate(args, model, eval_examples)

            def do_predict(args):

                from .ner_utils import save_ner_preds
                args.model_path = args.best_model_path
                test_examples = load_test_examples(args, test_data_generator)
                model = self.load_model()
                trainer.predict(args, model, test_examples)
                reviews_file, category_mentions_file = save_ner_preds(
                    args, trainer.pred_results, test_examples)
                return reviews_file, category_mentions_file

            if args.do_train:
                do_train(args)

            elif args.do_eval:
                do_eval(args)

            elif args.do_predict:
                do_predict(args)

            elif args.do_experiment:
                import mlflow
                from .utils import log_global_params
                if args.tracking_uri:
                    mlflow.set_tracking_uri(args.tracking_uri)
                mlflow.set_experiment(args.experiment_name)

                with mlflow.start_run(run_name=f"{args.local_id}") as mlrun:
                    log_global_params(args, self.experiment_params)

                    # ----- Train -----
                    do_train(args)

                    # ----- Predict -----
                    do_predict(args)

                    # ----- Submit -----
                    do_submit(args)
