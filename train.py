import os
# os.environ ['CUDA_VISIBLE_DEVICES'] = '2'
import shutil
import argparse
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from BaselineModel.utils.misc import BlackHole, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from BaselineModel.utils.train import *
from BaselineModel.models.dg_model import DG_Network
from BaselineModel.utils.mixed_dataloader import MixedDatasetManager, per_complex_corr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    num_cvfolds = config.train.num_cvfolds
    logdir = config.data.logdir
    
    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_dir = None
    else:
        if args.resume:
            log_dir = get_new_log_dir(logdir, prefix='%s-resume' % (config_name), tag=args.tag)
        else:
            log_dir = get_new_log_dir(logdir, prefix='%s' % (config_name), tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Model, Optimizer & Scheduler
    logger.info('Building model...')
    cv_mgr = CrossValidation(
        model_factory=DG_Network,
        config=config, 
        num_cvfolds=num_cvfolds
    ).to(args.device)
    
    # Data
    logger.info('Loading datasets...')
    dataset_mgr = MixedDatasetManager(
        config, 
        num_cvfolds=num_cvfolds, 
        num_workers=args.num_workers,
        logger=logger,
    )
    
    logger.info('Saving datasets complex...')
    with open(os.path.join(log_dir, 'train_pdb.pkl'), 'wb') as f:
        pickle.dump(dataset_mgr.train_split, f)
    with open(os.path.join(log_dir, 'val_pdb.pkl'), 'wb') as f:
        pickle.dump(dataset_mgr.val_split, f)

    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        cv_mgr.load_state_dict(ckpt['model'], )
    
    model, _, _ = cv_mgr.get(0)
#     print(model)
    
    def train(it):
        fold = it % num_cvfolds
        model, optimizer, scheduler = cv_mgr.get(fold)

        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(dataset_mgr.get_train_iterator(fold)), args.device)

        # Forward pass
        loss_dict, _ = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()
        
        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        if it%10==0:
            scalar_dict = {}
            scalar_dict.update({
                'fold': fold,
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })
            log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger, writer=writer)

    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        results = []
        with torch.no_grad():
            for fold in range(num_cvfolds):
                model, optimizer, scheduler = cv_mgr.get(fold)
                for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True)):
                    # Prepare data
                    batch = recursive_to(batch, args.device)

                    # Forward pass
                    loss_dict, output_dict = model(batch)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                    for complex, mutstr, dg_true, dg_pred in zip(batch['complex'], batch['mutstr'], output_dict['dG_true'], output_dict['dG_pred']):
                        results.append({
                            'complex': complex,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'dG': dg_true.item(),
                            'dG_pred': dg_pred.item()
                        })
        
        results = pd.DataFrame(results)
        if ckpt_dir is not None:
            results.to_csv(os.path.join(ckpt_dir, f'results_{it}.csv'), index=False)
        pearson_all = results[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
        spearman_all = results[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
        pearson_pc, spearman_pc = per_complex_corr(results)

        logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')
        logger.info(f'[PC]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f}')
        writer.add_scalar('val/all_pearson', pearson_all, it)
        writer.add_scalar('val/all_spearman', spearman_all, it)
        writer.add_scalar('val/pc_pearson', pearson_pc, it)
        writer.add_scalar('val/pc_spearman', spearman_pc, it)

        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(it, 'val', logger=logger, writer=writer)
        # Trigger scheduler
        for fold in range(num_cvfolds):
            _, _, scheduler = cv_mgr.get(fold)
            if it != it_first:  # Don't step optimizers after resuming from checkpoint
                if config.train.scheduler.type == 'plateau':
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': cv_mgr.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
