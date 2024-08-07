

import numpy as np
import pandas as pd
import os
import torch
from torch import nn


from src.learner_0shot import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import * 

import argparse

os.environ['CUDA_VISIBLE_DEVICES']='4'

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=720, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=64, help='patch length')
parser.add_argument('--stride', type=int, default=64, help='stride between patch')
parser.add_argument('--n_embedding', type=int, default=128, help='embedding size')
parser.add_argument('--L1_loss', type=int, default=1, help='use L1_loss')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=30, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default='zero-shot', help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='mfm+register', help='for multivariate model or univariate model')
parser.add_argument('--finetune_percentage', type=float, default=1, help='percentage of the train_set')
parser.add_argument('--one_channel', type=int, default=0, help='choose 1 channel')
parser.add_argument('--freeze_embedding', type=int, default=1, help='freeze the embedding layer')

args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/' + args.model_type + '/'
args.pretrain_path = 'saved_models/' + 'bigmodel' + '/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

suffix_name = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune)  + '_is_half'+ str(args.finetune_percentage) +'freeze_embedding'+str(args.freeze_embedding)+'_model' + str(args.finetuned_model_id)
if args.is_finetune: args.save_finetuned_model = args.dset_finetune+'_rose_finetuned'+suffix_name
elif args.is_linear_probe: args.save_finetuned_model = args.dset_finetune+'_rose_linear-probe'+suffix_name
else: args.save_finetuned_model = args.dset_finetune+'_rose_finetuned'+suffix_name

# get available GPU devide
set_device()

def get_model():

    weight_path = args.pretrain_path + args.pretrained_model + '.pt'
    model = torch.jit.load(weight_path)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model

def test_func():
    # get dataloader
    dls = get_dls(args)
    model = get_model().to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs,n_embedding=args.n_embedding,target_points=args.target_points)
    out  = learn.test(dls.test, scores=[mse,mae])         # out: a list of [pred, targ, score]
    print('score:', out[2])
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out


if __name__ == '__main__':

    mse_list =np.array([])
    mae_list =np.array([])
    args.dset = args.dset_finetune
    out = test_func()        
    print('----------- Complete! -----------')
            
    print(f'mse_mean:{mse_list.mean()},mae_mean:{mae_list.mean()}')
    result_mean = np.array([mse_list.mean(),mae_list.mean()])
    pd.DataFrame(result_mean.reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc_mean.csv', float_format='%.6f', index=False)
        


