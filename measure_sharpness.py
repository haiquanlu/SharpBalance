from __future__ import print_function
from typing import Dict, List, NamedTuple, Optional, Tuple
import argparse
import os
import sys
import shutil
import time
import random
import numpy as np
from tqdm import tqdm 
import pickle
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from pathlib import Path
import operator as op
from contextlib import contextmanager
import copy
from copy import deepcopy
from functorch import make_functional_with_buffers, vmap, grad
from sharpness_utils import eval_APGD_sharpness, eval_average_sharpness
from models.resnet_width import ResNet18,  ResNet50,  ResNet34, LogitNormalizationWrapper
from models.resnet_imgnet import resnet18

sys.path.append('../')
import utils




parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='wrn',
    choices=['wrn', 'allconv', 'densenet', 'resnext','resnet18','resnet34', 'wrn-28-10'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument(
    '--lr',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--norm',
    type=str)
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval_batch_size', type=int, default=5)
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
parser.add_argument('--width', default=64, type=int, help='Widen factor')
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
    '--pruned_model',
    default=False,
    type=bool
)
parser.add_argument('--adaptive', type=str, default='True') #action='store_true', 
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
parser.add_argument(
    '--mask_resumes',
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
    "--mode",
    type=str
)
parser.add_argument(
    '--sam_epochs',
    type=int,
    default=15,
    help=''
)
parser.add_argument(
    '--tryy',
    type=int,
    default=0,
    help=''
)
parser.add_argument(
    '--gamma',
    type=str,
    default='0.5',
    help=''
)
parser.add_argument(
    '--sam_iter',
    type=int,
    default=1
)
parser.add_argument(
    '--retrain_epochs',
    type=int,
    default=20,
    help=''
)
parser.add_argument(
    '--retrain_iter',
    type=int,
    default=1
)
parser.add_argument(
    '--augmix',
    type=bool,
    default=False
)
parser.add_argument(
    '--flat_ratio',
    type=float,
    default=0.2
)
parser.add_argument(
    '--compare_ratio',
    type=float,
    default=0.2
)
parser.add_argument(
    '--lrsche',
    type=str,
    default='constant'
)
parser.add_argument(
    '--optimizer',
    type=str,
    default='SGD'
)
parser.add_argument(
   '--only_flat',
   type=str,
   default='False'
)
parser.add_argument(
    '--flat_trial',
    default=5,
    type=int)
parser.add_argument(
    '--no_aug',
    default=True,
    type=bool)
parser.add_argument(
    '--batch-num',
    type=int,
    default=100
)
parser.add_argument(
    '--logit-norm',
    type=str,
    default='False'
)


parser.add_argument("--train_idx_path",type=str, default='./dataset/train_idx.npy')
parser.add_argument("--val_idx_path",type=str, default='./dataset/valid_idx.npy')
parser.add_argument("--data_path",type=str)

parser.add_argument('--mod',type=int,default=0)
                    
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global fmodel
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __getitem__(self, i):
        x, y = self.dataset[i]
        return i,x,y
    def __len__(self):
        return len(self.dataset)

def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)

def test_accuracy(model,test_loader):
    model.eval()
    n = 0
    accuracy=torch.zeros(len(test_loader.dataset))
    with torch.no_grad():
        for idx, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
    
            pred = output.argmax(dim=1)
            correct= pred.eq(target).float().cpu()
            accuracy[idx]=correct
    return accuracy

@contextmanager
def _perturbed_model(
  model,
  sigma: float,
  rng,
  magnitude_eps: Optional[float] = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model
def compute_grad(model,sample, target):
    model.eval()
    model.zero_grad()
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)
    prediction = model(sample)
    loss_fn=nn.CrossEntropyLoss()
    
    loss = loss_fn(prediction, target)
    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(model,data, targets):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(model,data[i], targets[i]) for i in range(data.shape[0])]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads
@torch.no_grad()
def pac_bayes(net,train_data,accuracy,seed,  
    magnitude_eps: Optional[float] = None,
    search_depth: int = 15,
    montecarlo_samples: int = 10,
    accuracy_displacement: float = 0.1,
    displacement_tolerance: float = 1e-2):

    train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True)

    BIG_NUMBER = 10348628753
    device = next(net.parameters()).device
    rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
    rng.manual_seed(BIG_NUMBER + seed)

    flatness=torch.zeros(len(train_loader.dataset))
    for i, dt in enumerate(tqdm(train_loader)):
        idx,data,target=dt
        lower, upper = 0, 2
        for _ in range(search_depth):
            sigma=(lower+upper)/2
            accuracy_samples = []
            for _ in range(montecarlo_samples):
                with _perturbed_model(net, sigma, rng, magnitude_eps) as p_model:
                    data, target = data.to(device, dtype=torch.float), target.to(device)
                    logits = p_model(data)
                    pred = logits.argmax(dim=1)
                    correct= pred.eq(target).float().cpu()
                    loss_estimate = correct.sum()
                    accuracy_samples.append(loss_estimate)
            displacement=abs(np.mean(accuracy_samples) - accuracy[idx])
            if abs(displacement - accuracy_displacement) < displacement_tolerance:
                break
            elif displacement > accuracy_displacement:
                # Too much perturbation
                upper = sigma
            else:
                # Not perturbed enough to reach target displacement
                lower = sigma
        flatness[idx]=sigma
    return flatness.numpy()

'''
def forgot(model_lst,train_loader,train_data):
    for iter in range(args.retrain_iter):
'''
def sam_idx(save_path,flats,i,ascend=True):
    print(i) 
    indice=np.arange(flats[0].shape[0])
    if(args.compare_ratio!=0):
        total=int(indice.shape[0]*args.compare_ratio)
        for j in range(len(flats)):

            flat=flats[j]
            indice1=np.argsort(flat)
            if(ascend):
                indice=np.intersect1d(indice,indice1[:total])
            else:
                indice=np.intersect1d(indice,indice1[-total:])
        print(indice.shape)
        flat_intection=np.zeros((len(flats),indice.shape[0]))
        for k in range(len(flats)):
            flat_intection[k]=flats[k][indice]           
        static=[]
        this_flats=np.array(flats)
        np.save(save_path,this_flats)        
        this_indice=indice[i::len(flats)]
        return np.array(list(set(indice)-set(this_indice)))
    else:
        return np.array([-1])



def mask_gradient(net,mask):
    for name, tensor in net.named_parameters():
        #print(name)
        if name in mask:
            #print(name)
            tensor.grad = tensor.grad*mask[name]
            
            
            
def test_worstcase(net,train_loader,mask):
    net.eval()
    total_loss=0
    base_optimizer = torch.optim.SGD
    optimizer = NSAM(net.parameters(),base_optimizer,rho=args.gamma,lr=0.1,momentum=0.9,weight_decay=5e-4)     
    flatness=torch.zeros(len(train_loader.dataset))
    for i,dt in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        idx,data,target=dt
        data, target = data.cuda(), target.cuda()
        logits = net(data)
        
        loss = F.cross_entropy(logits, target)
        total_loss+=loss
        loss.backward()
        mask_gradient(net,mask)
        optimizer.first_step()
        
        output=net(data)
        rloss = F.cross_entropy(output, target)
        sharpness=rloss-loss
        flatness[idx]=sharpness.detach()
        
        optimizer.back_step()
    return flatness.numpy()
        

def hessian_ensemble(deck,masks,train_loader,savepaths):

    deck.eval() 
    flatness=torch.zeros(len(train_loader.dataset))
    criterion = nn.CrossEntropyLoss()
    for i, dt in enumerate(tqdm(train_loader)):

        idx,data,target=dt
        if(args.eval_batch_size==1 and idx%10!=args.mod):
            continue      
        hessian_dataloader = (data, target)
        
        hessian_comp = hessian(deck,
                        criterion,
                        data=hessian_dataloader,
                        cuda=True)
        trace = hessian_comp.trace()
        print('\n***Trace: ', np.mean(trace))
        flatness[idx]=(np.mean(trace))
    Path(savepaths).mkdir(parents=True, exist_ok=True)
    path=os.path.join(savepaths,'hessian_ensemble_largebatch.npy')
    np.save(path,flatness)
    return


def apply_mask_graient_deck(deck,ft_per_sample_grads,masks):
    i=0
    for name,param in deck.named_parameters():
        index=int(name[7])
        mask=masks[index]
        if name[9:] in mask.masks:
            ft_per_sample_grads[i]=ft_per_sample_grads[i]*torch.unsqueeze(mask.masks[name[9:]],dim=0)
        i=i+1
    return ft_per_sample_grads
            

def test_hessian(net, train_loader, mask, mode='trace'):
    net.eval() 
    flatness=[]
    hessian_dataloader = []
    criterion = nn.CrossEntropyLoss()
    for i, dt in enumerate(tqdm(train_loader)):
        if(i == args.batch_num):
            break
        idx,data,target=dt
        hessian_dataloader.append((data, target))
    
    hessian_comp = hessian(net,
                    criterion,
                    mask,
                    dataloader=hessian_dataloader,
                    cuda=True)
        
    if mode == 'trace':
        trace = hessian_comp.trace()        
        flatness.append((np.mean(trace)))
        print('\n***Trace: ', np.mean(trace))
    elif mode == 'top_e':
        top_eigenvalues, _ = hessian_comp.eigenvalues()
        flatness.append(top_eigenvalues)
        print('\n***top_eigenvalues: ', np.mean(top_eigenvalues))
    
    flatness=np.array(flatness)
    return flatness

def hessians(model_lst, savepaths, masks, mode='trace'):
    nets_flatness=[]
    for i in range(len(model_lst)):
        train_loader,_ = get_trainloader()
        net=model_lst[i]
        if(len(masks)>0):
            mask=masks[i].masks
        else:
            mask=None
        flatness=test_hessian(net, train_loader, mask, mode)
        nets_flatness.append(flatness)
        
    savepaths = os.path.join(savepaths, f'bs{args.eval_batch_size}_bn{args.batch_num}')
    Path(savepaths).mkdir(parents=True, exist_ok=True)
    path=os.path.join(savepaths,f'hessian_{mode}_indiv_largebatch.npy')
    np.save(path,nets_flatness)
    
    
def get_ntk(networks, savepaths, masks):
    train_loader,_ = get_trainloader()
    log_det_ntk_over_data = []
    for i, dt in enumerate(train_loader):
        if i == args.batch_num:
            break
        _, inputs, targets = dt
        inputs = inputs.cuda(non_blocking=True)
        log_det_ntk_lst = []
        for network in networks:
            network.eval()
            grads = []

            if len(masks)>0:
                sparse_model=True
            else:
                sparse_model=False

            network.zero_grad()
            inputs_ = inputs.clone().cuda(non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits

            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        # the model is a sparse model
                        if sparse_model:
                            grad.append(W.grad[W != 0].view(-1).detach())
                        else:
                            #rint('Dense Model')
                            grad.append(W.grad.view(-1).detach())
                grads.append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()

            grads = torch.stack(grads, 0) 
            # print("stacking matrices")
            print("grads.shape", len(grads), grads.shape)
            # print("computing the matrix multiplication")
            ntk = torch.einsum('nc,mc->nm', [grads, grads])
            # print("matrix multiplication completed")
            print(len(ntk), ntk.shape)

            eigenvalue = torch.linalg.eigvalsh(ntk)
            log_det_ntk_lst.append(np.log(eigenvalue.detach().cpu().numpy()).sum())

            print(len(eigenvalue), log_det_ntk_lst)
        log_det_ntk_over_data.append(log_det_ntk_lst)

    log_det_ntk_over_data = np.array(log_det_ntk_over_data)
    savepaths = os.path.join(savepaths, f'bs{args.eval_batch_size}_bn{args.batch_num}')
    Path(savepaths).mkdir(parents=True, exist_ok=True)
    path=os.path.join(savepaths,f'log_det_ntk.npy')
    print(f'--------------------> save to {path}')
    np.save(path, log_det_ntk_over_data)
    


def get_trainloader():
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    print(f"Random seed set as {args.seed}")
    if args.dataset=='cifar10':
        mean=[0.4914,0.4822,0.4465]
        std=[0.2470, 0.2435, 0.2616]
    else:
        mean=[0.5071, 0.4865, 0.4409]
        std=[0.2673, 0.2564, 0.2762]

    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transform = preprocess
    train_idx=np.load(args.train_idx_path)
    val_idx= np.load(args.val_idx_path)
    if args.dataset == 'cifar10':
        train_data = torch.utils.data.Subset(datasets.CIFAR10(
                    f'{args.data_path}/cifar10', train=True, 
                    transform=test_transform, download=True),indices=train_idx)
        test_data = datasets.CIFAR10(
                    f'{args.data_path}/cifar10', train=False, 
                    transform=test_transform, download=True)
        base_c_path = f'{args.data_path}/CIFAR-10-C/'
        num_classes = 10
    else:
        train_data = torch.utils.data.Subset(datasets.CIFAR100(
                    f'{args.data_path}/cifar100', train=True, 
                    transform=test_transform, download=True),indices=train_idx)

        test_data = datasets.CIFAR100(
                    f'{args.data_path}/cifar100', train=False, transform=test_transform, download=True)
        base_c_path = f'{args.data_path}/CIFAR-100-C/'
        num_classes = 100  

    train_data=CustomDataset(train_data)
    print('train_data without augmix')

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    return train_loader, test_loader


def APGD(model_lst, savepaths, masks, mode):
    nets_flatness=[]
    loss_f = lambda logits, y: F.cross_entropy(logits, y, reduction='mean')
    for i in range(len(model_lst)):
        net=model_lst[i]
        net.eval()
        net = LogitNormalizationWrapper(net, normalize_logits=args.logit_norm == 'True')
        if(len(masks)>0):
            mask=masks[i].masks
        else:
            mask=None
        
        sharpness_ablation = {}
        #, 
        for norm in ['l2', 'linf']:
            for gamma in args.gamma:
                train_loader,_ = get_trainloader()
                args.norm = norm 
                if mask is None:
                    print("--------------------> Evaluate a dense model")
                    print(f'{mode}, args.logit_norm={args.logit_norm} rho={gamma}, adaptive={args.adaptive == "True"}, {args.norm}, num_batch={args.batch_num}, train_loader batch={train_loader.batch_size}')
                    if mode == 'APGD_worst':
                        flatness=eval_APGD_sharpness(net,
                                                    mask,
                                                    train_loader,
                                                    loss_f=loss_f,
                                                    train_err=None, 
                                                    train_loss=None,
                                                    n_iters=20,
                                                    rho=gamma, 
                                                    adaptive= args.adaptive == 'True',
                                                    norm=args.norm,
                                                    T=args.batch_num)
                    elif mode == 'APGD_average':
                        flatness=eval_average_sharpness(net,
                                                    mask,
                                                    train_loader,
                                                    loss_f=loss_f,
                                                    n_iters=100,
                                                    T=args.batch_num,
                                                    rho=gamma, 
                                                    adaptive= args.adaptive == 'True',
                                                    norm=args.norm,
                                                    )
                        
                else:
                    print("--------------------> Evaluate a sparse model")
                    print(f'{mode}, args.logit_norm={args.logit_norm} rho={gamma}, adaptive={args.adaptive == "True"}, {args.norm}, num_batch={args.batch_num}, {train_loader.batch_size}')
                    if mode == 'APGD_worst':
                        flatness=eval_APGD_sharpness_sparse(net,
                                                            mask,
                                                            train_loader,
                                                            loss_f=loss_f,
                                                            train_err=None,
                                                            train_loss=None,
                                                            n_iters=20,
                                                            rho=gamma, 
                                                            adaptive=args.adaptive == 'True',
                                                            norm=args.norm,
                                                            T=args.batch_num)
                    elif mode == 'APGD_average':
                        flatness=eval_average_sharpness_sparse(net,
                                                            mask,
                                                            train_loader,
                                                            loss_f=loss_f,
                                                            n_iters=100,
                                                            T=args.batch_num,
                                                            rho=gamma, 
                                                            adaptive= args.adaptive == 'True',
                                                            norm=args.norm,
                                                            )
                sharpness_ablation[f'{args.norm}_{gamma}'] = flatness
                print(sharpness_ablation)
        
        nets_flatness.append(sharpness_ablation)
        
    savepaths = os.path.join(savepaths, f'bs{args.eval_batch_size}_bn{args.batch_num}')
    Path(savepaths).mkdir(parents=True, exist_ok=True)
    path=os.path.join(savepaths,   f'{mode}_adaptive{args.adaptive}_logitnorm{args.logit_norm}_indiv.pkl')
    f=open(path,"wb")
    pickle.dump(nets_flatness,f)  
    f.close()

def compute_loss_stateless_model (params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    loss_fn=nn.CrossEntropyLoss()
    global fmodel
    predictions = fmodel(params, buffers, batch) 
    loss = loss_fn(predictions, targets)
    return loss


def apply_mask_graient(net,ft_per_sample_grads,mask):
    i=0
    for name, param in net.named_parameters():
        if name in mask:
            ft_per_sample_grads[i]=ft_per_sample_grads[i]*torch.unsqueeze(mask[name],dim=0)
        i=i+1
    return ft_per_sample_grads

def test_fisher(model_lst, savepaths, masks):
    flatness_model_lst = []

    for model_i in range(len(model_lst)):
        net = model_lst[model_i]
        net.eval()
        train_loader, _ = get_trainloader()
        if(len(masks)>0):
            mask=masks[model_i].masks
        else:
            mask=None
        flatness=[] 
        global fmodel
        fmodel, params, buffers = make_functional_with_buffers(net)
        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

        for i, dt in enumerate(tqdm(train_loader)):
            if i == args.batch_num:
                break
            idx,data,target=dt
            
            #print(idx)
            data,target=data.to(device, dtype=torch.float), target.to(device)
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, target)
            ft_per_sample_grads=list(ft_per_sample_grads)
            if mask is not None:
                ft_per_sample_grads=apply_mask_graient(net,ft_per_sample_grads,mask)
                
            for j in range(len(ft_per_sample_grads)):
                ft_per_sample_grads[j]=ft_per_sample_grads[j].view(data.shape[0],-1)
            
            grads=torch.cat(ft_per_sample_grads,dim=1)
            fisher=grads.pow(2).sum(dim=1)
            
            flatness.append(fisher.cpu().detach())
        flatness = torch.cat(flatness).numpy()
        print(flatness.shape)
        flatness_model_lst.append(np.mean(flatness))
    
    flatness_model_lst = np.array(flatness_model_lst)
    savepaths = os.path.join(savepaths, f'bs{args.eval_batch_size}_bn{args.batch_num}')
    Path(savepaths).mkdir(parents=True, exist_ok=True)
    path=os.path.join(savepaths,  'fisher.npy')
    np.save(path,flatness_model_lst)

def main():

    if args.gamma:
        args.gamma = [float(item) for item in args.gamma.split(',')]
        print('args.gamma', args.gamma)
    else:
        print('No numbers provided')
    
    if_sparse = False
    if args.dataset == 'cifar10':

        num_classes = 10
    else:
        num_classes = 100  



    # Distribute model across all visible GPUs
    cudnn.benchmark = True

    net_ensemble_lst = []
    masks=[]
    savepaths=[]

    i=0
    if args.resume:
        if len(args.resume) > 0:
            for file in args.resume:
                print(file)
                if args.model == 'densenet':
                    print("hh")
                elif 'wrn' in args.model:
                    print(args.layers, args.width)
                    net = WideResNet(args.layers, args.width , num_classes, args.droprate)  
                elif args.model=='resnet18':
                    net=ResNet18(width=args.width,num_classes=num_classes)
                elif args.model=='resnet34':
                    net=ResNet34(width=args.width,num_classes=num_classes)
                elif args.model=='resnet50':
                    net=ResNet50(width=args.width,num_classes=num_classes)

                net=net.cuda()
                indice=file.rindex('/')
                savepath=file[:indice]
                savepaths.append(savepath)
                print(savepath)
                
                checkpoint=torch.load(file,map_location='cuda:0')     
                if 'mask' in checkpoint:              
                    mask=checkpoint['mask']
                    masks.append(mask)
                net.load_state_dict(checkpoint['state_dict'])
                if ('mask' in checkpoint and  args.mode=='hessian'):
                    for name,module in net.named_modules():
                        if(name+'.weight') in mask.masks:
                            print(name)
                            prune.custom_from_mask(module,name='weight',mask=mask.masks[name+'.weight'])
                print(f'Model restored from: {file} ')
                i=i+1
                net.eval()
                net_ensemble_lst.append(net)


    if args.evaluate:
        if(args.mode=='hessian_t'):
            print("--------------------> enter hessian trace")
            hessians(net_ensemble_lst       ,args.save,  masks, mode='trace')
        elif (args.mode=='hessian_e'):
            print("--------------------> enter hessian Eigen")
            hessians(net_ensemble_lst       ,args.save,masks , mode='top_e')
        elif (args.mode=='fisher'):
            print("--------------------> enter Fisher")
            test_fisher(net_ensemble_lst       ,args.save,masks)
        elif('APGD' in args.mode):
            APGD(net_ensemble_lst   , args.save, masks, mode=args.mode)
        elif(args.mode=='ntk'):
            get_ntk(net_ensemble_lst,     args.save, masks)
            

if __name__ == '__main__':
    main()
