from __future__ import print_function

import argparse
import os
import sys
import shutil
import time
import random
import numpy as np
import pickle
import utils

from models.resnet_width import ResNet18
from models.resnet_imgnet import resnet18

from cka_utils import *

import torch.nn as nn
import torch 
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from pathlib import Path
from torch.utils.data import ConcatDataset


parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'tiny_imgnet'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='wrn',
    choices=['wrn', 'allconv', 'densenet', 'wrn-28-10', 'resnext','resnet18','resnet50','resnet34'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=18, type=int, help='total number of layers') # 
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    nargs='+',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', type=bool, default=True, help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=6,
    help='Number of pre-fetching threads.')
parser.add_argument(
    '--seed',
    type=int,
    default=13,
    help='')
parser.add_argument(
    '--width',
    type=int,
    default=64,
    help='')
parser.add_argument(
   "--mode",
   type=str
)
parser.add_argument(
   "--mgpu",
   default=False,
   type=bool
)
parser.add_argument(
   "--averaging_weight",
   default=False,
   type=bool
)
parser.add_argument(
   "--averaging_num",
   default=3,
   type=int
)
parser.add_argument(
   "--no_aug",
   default=True,
   type=bool
)
parser.add_argument("--train_idx_path",type=str, default='./dataset/train_idx.npy')
parser.add_argument("--val_idx_path",type=str, default='./dataset/valid_idx.npy')
parser.add_argument("--data_path",type=str)

parser.add_argument("--with-data-augmentation", action='store_true', default=False)
parser.add_argument("--CKA-batches", type=int, default=10, help='number of batches for computing CKA')
parser.add_argument("--CKA-repeat-runs", type=int, default=1, help='number of repeat for CKA')
parser.add_argument('--flattenHW', default = False, action = 'store_true', help = 'flatten the height and width dimension while only comparing the channel dimension')
parser.add_argument('--not-input', dest='not_input', default = False, action='store_true', help='no CKA computation on input data')
parser.add_argument('--mixup-CKA', dest='mixup_CKA', default = False, action='store_true', 
                            help='measure CKA on mixup data')


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))



def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    nll_loss = 0
    correct = 0
    n = 0
    max_prob=[]
    corrects=[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            nll_loss = F.nll_loss(torch.log(softmax_preds), target, reduction='mean').item()  # NLL
            test_loss += F.nll_loss(torch.log(softmax_preds), target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            max_probability=torch.max(softmax_preds,dim=1)[0].cpu().numpy()
            correctness=pred.eq(target.view_as(pred)).cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]
            max_prob.append(max_probability)
            corrects.append(correctness)
    test_loss /= float(n)
    print(f"correct:{correct / float(n)},loss:{test_loss}")
    return correct / float(n), test_loss, max_prob, corrects

def compute_bias_variance(net, testloader, trial, num_classes, OUTPUST_SUM, OUTPUTS_SUMNORMSQUARED):
    net.eval()
    bias2 = 0
    variance = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_onehot = torch.FloatTensor(targets.size(0), num_classes).cuda()
            targets_onehot.zero_()

            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            
            outputs = net(inputs)
            OUTPUST_SUM[total:(total + targets.size(0)), :] += outputs
            OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)] += outputs.norm(dim=1) ** 2.0

            bias2 += (OUTPUST_SUM[total:total + targets.size(0), :] / (trial + 1) - targets_onehot).norm() ** 2.0
            variance += OUTPUTS_SUMNORMSQUARED[total:total + targets.size(0)].sum()/(trial + 1) - (OUTPUST_SUM[total:total + targets.size(0), :]/(trial + 1)).norm() ** 2.0
            total += targets.size(0)

    return bias2 / total, variance / total

def variance_c(model_lst, num_classes, OUTPUST_SUM, OUTPUTS_SUMNORMSQUARED, testloader):
    trial = 0
    total_trial = len(model_lst)
    variance_unbias_lst = []
    for net in model_lst:
        bias, variance = compute_bias_variance(net, testloader, trial, num_classes, OUTPUST_SUM, OUTPUTS_SUMNORMSQUARED)
        variance_unbias = variance * total_trial / (total_trial - 1.0)
        
        variance_unbias_lst.append(variance_unbias.item())
        trial += 1
        
    return variance_unbias_lst


def main():
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    print(f"Random seed set as {args.seed}")
    print(args)

    if args.dataset=='cifar10':
        mean=[0.4914,0.4822,0.4465]
        std=[0.2470, 0.2435, 0.2616]
    elif args.dataset=='cifar100':
        mean=[0.5071, 0.4865, 0.4409]
        std=[0.2673, 0.2564, 0.2762]
    else:
        mean=[0.480, 0.448, 0.397]
        std=[0.276, 0.269, 0.282]
    
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    
    
    
    test_transform = preprocess
    train_idx=np.load(args.train_idx_path)
    val_idx= np.load(args.val_idx_path)
    if args.dataset == 'cifar10':
        train_data = torch.utils.data.Subset(datasets.CIFAR10(
                    os.path.join(args.data_path, 'cifar10'), train=True, transform=test_transform, download=True),indices=train_idx)
        test_data = datasets.CIFAR10(
                    os.path.join(args.data_path, 'cifar10'), train=False, transform=test_transform, download=True)
        base_c_path = args.data_path + '/' + 'CIFAR-10-C/'
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = torch.utils.data.Subset(datasets.CIFAR100(
                    os.path.join(args.data_path, 'cifar100'), train=True, transform=test_transform, download=True),indices=train_idx)
        test_data = datasets.CIFAR100(
                        os.path.join(args.data_path, 'cifar100'), train=False, transform=test_transform, download=True)
        base_c_path =  args.data_path + '/' + 'CIFAR-100-C/'
        num_classes = 100
    else:
        train_data = datasets.ImageFolder(args.data_path+'/tiny-imagenet-200/train', test_transform)
        val_data = datasets.ImageFolder(args.data_path+'/tiny-imagenet-200/val', test_transform)
        test_data = datasets.ImageFolder(args.data_path+'/tiny-imagenet-200/val', test_transform)
        num_classes = 200
        base_c_path =  args.data_path + '/' + 'Tiny-ImageNet-C'
    
    # Distribute model across all visible GPUs
    cudnn.benchmark = True

    net_ensemble_lst = []

    if len(args.resume) > 0:
        for file in args.resume:
            # Create model
            if args.model=='resnet18':
                if args.dataset == 'tiny_imgnet':
                    net = resnet18(args.width,num_classes=num_classes)
                else:
                    net=ResNet18(width=args.width,num_classes=num_classes)
            
            checkpoint = torch.load(file,map_location='cuda:0')


            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            net_ensemble_lst.append(net)
            net = torch.nn.DataParallel(net).cuda()
    
            print(f'Model restored from: {file}')
    


    if args.evaluate:
        if 'variance_mse_c' in args.mode:
            if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                Y_ood_list = []
                for index_c in range(len(CORRUPTIONS)):
                    if index_c == 0:
                        print(f"index_c: {index_c}", np.load(f'{base_c_path}/{CORRUPTIONS[index_c]}.npy').shape)
                        X_ood = np.load(f'{base_c_path}/{CORRUPTIONS[index_c]}.npy')[:50000]
                    else:
                        print(f"index_c: {index_c}", np.load(f'{base_c_path}/{CORRUPTIONS[index_c]}.npy').shape)
                        X_ood_idx = np.load(f'{base_c_path}/{CORRUPTIONS[index_c]}.npy')[:50000]
                        X_ood = np.concatenate((X_ood, X_ood_idx), axis=0)
                        
                    Y_ood = np.load(f'{base_c_path}/labels.npy')[:50000]
                    for i in range(50000):
                        Y_ood_list.append(Y_ood[i])   

                             
                print('X (OOD) shape: ', X_ood.shape)
                print('Y (OOD) shape: ', len(Y_ood_list))
                
                test_data.data = X_ood
                test_data.targets = Y_ood_list
                testloader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False, num_workers=6)
            else:
                for index_c in range(len(CORRUPTIONS)):
                    for s in range(1, 6):
                        if index_c == 0 and s == 0:
                            valdir = os.path.join(base_c_path, str(s))
                            corr_set = datasets.ImageFolder(valdir, test_transform)[:50000]
                            corr_dataset = corr_set
                        else:
                            valdir = os.path.join(base_c_path, str(s))
                            corr_set = datasets.ImageFolder(valdir, test_transform)[:50000]
                            corr_dataset = ConcatDataset(corr_dataset, corr_set)

                testloader = torch.utils.data.DataLoader(corr_dataset, batch_size=500, shuffle=False, num_workers=6)
            
            test_size = len(test_data.targets)
            OUTPUST_SUM = torch.Tensor(test_size, num_classes).zero_().cuda()
            OUTPUTS_SUMNORMSQUARED = torch.Tensor(test_size).zero_().cuda()
            variance_unbias_lst = variance_c(net_ensemble_lst, num_classes, OUTPUST_SUM, OUTPUTS_SUMNORMSQUARED, testloader) #model_lst, num_classes, OUTPUST_SUM, OUTPUTS_SUMNORMSQUARED

            Path(args.save).mkdir(parents=True, exist_ok=True)
            path=os.path.join(args.save, f'variance_c_mse.pkl')
            print(path)
            f=open(path,"wb")
            pickle.dump(variance_unbias_lst, f)
            f.close()
            
    
    

if __name__ == '__main__':
    main()