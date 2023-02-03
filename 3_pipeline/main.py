import sys
import argparse
import time
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.optim.lr_scheduler import StepLR

from gpl.utils.evaluate import log_metrics


@hydra.main(version_base=None, config_path='../0_config', config_name='config')
def main(args: DictConfig):
    from gpl import LOG_DIR, TBLOG_DIR, TBLOG_HPARAMS_DIR, CKPT_DIR, TMP_DIR
    from gpl.utils.utils import load_config
    from gpl.utils.evaluate import  prototype_performance 
    from gpl.utils.get_dataset import get_dataset, get_dataloaders
    from gpl.utils.get_model import get_model
    from gpl.training import Trainer, DataLoaders, get_optimizer, get_default_logger
    
    EXP_NAME, maybe_altered_args = load_config(args)
    EXP_GROUP_NAME = maybe_altered_args.exp_group_name
    ################################################## using config, EXP_NAME, and EXP_GROUP_NAME
    
    # set wandb
    if args.train and not args.debug:
        wandb.init(
            project='GPL',
            name=EXP_NAME,
            config=args,
            group=EXP_GROUP_NAME,
            notes=args.wandb_note if args.wandb_note != '' else None
            )
        logger = get_default_logger(log_dir=LOG_DIR, exp_name=EXP_NAME, to_console=args.log.log_console, to_file=args.log.log_file, args={**vars(args), **args})
    else:
        logger = None
        pass
    
    ####### running code
    device = maybe_altered_args.device
    debug = maybe_altered_args.debug

    # dataset
    compute_degree = True if maybe_altered_args.model.name == 'PNA' else False
    train_set, val_set, test_set, extra_info = get_dataset(maybe_altered_args.dataset.name, splits=maybe_altered_args.dataset.splits, device=device, \
                                                            mutag_x=maybe_altered_args.dataset.mutag_x, compute_degree=compute_degree)
    print('Dataset info:', extra_info)

    data_loaders = get_dataloaders(train_set, val_set, test_set, batch_size=maybe_altered_args.dataset.batch_size)
    loaders_wraper = DataLoaders(data_loaders)

    with open_dict(maybe_altered_args):
        maybe_altered_args.framework.num_class = extra_info['num_class']
        maybe_altered_args.framework.multi_label = extra_info['multi_label']
   
    model_constructor = get_model
    model_constructor_params = {
        'x_dim': extra_info['x_dim'],
        'edge_attr_dim': extra_info['edge_attr_dim'],
        'num_class': extra_info['num_class'],
        'multi_label': extra_info['multi_label'],
        'deg': extra_info['deg'],
        'config': maybe_altered_args,
    }
    
    optimizer_params = maybe_altered_args['training']['optimizer_params']
    scheduler_params = maybe_altered_args['training']['scheduler_params']
    epochs = maybe_altered_args['training']['epochs']

    print('Experiment name:', EXP_NAME)
    
    hparams_save = maybe_altered_args
    
    train_epoch_callbacks = [log_metrics, ]
    evaluation_callbacks = [prototype_performance]

    # run
    trainer = Trainer(
        model_name=maybe_altered_args.model.name,
        dataset_name=maybe_altered_args.dataset.name,
        model_constructor=model_constructor,
        model_constructor_params=model_constructor_params,
        data_loaders=loaders_wraper,
        optimizer_constructor=get_optimizer,
        optimizer_params=optimizer_params,
        scheduler_constructor=StepLR,
        scheduler_params=scheduler_params,
        random_seed=maybe_altered_args.random_seed,
        logger=logger,
        tb_log_dir=TBLOG_DIR/EXP_GROUP_NAME,
        ckpt_dir=CKPT_DIR,
        log_dir=LOG_DIR,
        tmp_dir=TMP_DIR,
        hparams_save=hparams_save,
        hparams_save_dir=TBLOG_HPARAMS_DIR,
        epochs=epochs,
        device=device,
        train_epoch_callbacks=train_epoch_callbacks,
        debug=debug,
        experiment_name=EXP_NAME,
        task_id=maybe_altered_args.task_id,
        val=True,
        training_mode=args.train,
        log2tensorboard=False,
    )

    if args.train:
        trainer.train()
        wandb.finish()
    else:
        trainer.evaluate(ckpt_file=maybe_altered_args.evaluation.ckpt_file, evaluate_data=False, evaluation_callbacks=evaluation_callbacks)

    print(f'Finished, exp name: {EXP_NAME}')
    


if __name__ == "__main__":
    main()
