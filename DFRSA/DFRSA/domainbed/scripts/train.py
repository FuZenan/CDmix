# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import copy
import json
import os
import random
import sys
import time
import uuid
import sys
sys.path.append("D:/VScode/TMC/XDomainMix-main")

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default= "\\")
    parser.add_argument('--dataset', type=str, default="DSADS")
    parser.add_argument('--algorithm', type=str, default="CDmix",choices="CDmix,CDmix_sam,ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=42,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=2000,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        gpu_index = 0
        device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError
    
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=32,
        num_workers=1)
        for i, env in enumerate(dataset)
        if i not in args.test_envs]
    
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=1)
        for i, env in enumerate(dataset)
        if i not in args.test_envs]
    
    test_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=1)
        for i,env in enumerate(dataset)
        if i in args.test_envs or i in [j + 4 for j in args.test_envs]] 

    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))
        if i not in args.test_envs]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))
        if i not in args.test_envs]   
    test_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))
        if i in args.test_envs]
    test_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))
        if i in args.test_envs]   
   
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits]) # 30
    n_steps = args.steps or dataset.N_STEPS # 100 
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def cal_val_acc(test_key_name, result):
        total_acc = 0
        count = 0
        for key in test_key_name:
            total_acc += result[key+'_acc']
            count += 1
        return total_acc/count  # "env1_out", "env2_out", "env3_out" 计算这三的平均准确率

    last_results_keys = None

    best_accuracy = 0
    best_step = 0
    best_test_name = ""
    save_path = "best_model.pth"    
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                            for x, y, z in next(train_minibatches_iterator)]
        
        step_vals = algorithm.update(minibatches_device, None)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():  # loss_domain； loss_task
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            tests = zip(test_loader_names, test_loaders, eval_weights)

            total_correct = 0
            total_samples = 0
            for name, loader, weights in evals:
                print("eval_", name)
                acc, correct, total = misc.accuracys(
                    algorithm, loader, weights, device)
            for name, loader, weights in tests:  # 遍历测试数据
                print("test_", name)
                acc, correct, total = misc.accuracys(
                    algorithm, loader, weights, device) 

                results[name + '_acc'] = acc
                total_correct += correct
                total_samples += total

            total_accuracy = total_correct / total_samples
            print("Total Accuracy: {:.2f}%".format(100 * total_accuracy))

            if total_accuracy > best_accuracy:
                best_accuracy = total_accuracy
                best_step = step
                best_test_name = name

            torch.save(algorithm.state_dict(), save_path)

            print("Best Accuracy: {:.2f}% at step {} on test set {}".format(100 * best_accuracy, best_step, best_test_name))

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            if (args.algorithm == 'CDmix' and step >= hparams['warmup_step']):
                epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            with open(os.path.join(args.output_dir, 'done'), 'w') as f:
                f.write('done')