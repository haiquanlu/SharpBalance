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
from utils import create_dataset, create_model, CustomDataset, save_model, load_model
import torch.nn as nn
import torch 
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from pathlib import Path
from models.resnet import ResNet18, ResNet18_softmax
from models.resnet_imgnet import resnet18


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



def evaluate_ensemble(model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            current_fold_preds.append(softmax_preds)
            test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data

def test(model_lst):
    """Evaluate network on given dataset."""
    train_loader, val_loader, test_loader = get_trainloader()
    all_preds=[]
    val_acc = []
    nll_loss=[]
    for model in model_lst:
        indi_acc, indi_loss,max_prob,clean_correct  = evaluate(model, device, test_loader)
        val_acc.append(indi_acc)
        nll_loss.append(indi_loss)
        current_preds,target=evaluate_ensemble(model, device, test_loader)
        all_preds.append(current_preds)
        
    output_mean = torch.mean(torch.stack(all_preds, dim=0), dim=0)
    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    dic={}
    dic['ensemble_accuracy']=correct/float(n)
    dic['individual_accuracy']=val_acc
    dic['individual_loss']=nll_loss
    dic['ensemble_loss']=F.nll_loss(torch.log(output_mean), target, reduction='mean').item()
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_clean_accuracy.pkl')
    f=open(path,"wb")
    pickle.dump(dic,f)
    f.close()
    path2=os.path.join(args.save,"max_probs.npy")
    path3=os.path.join(args.save,"corrects.npy")
    np.save(path2,max_prob)
    np.save(path3,clean_correct)


def test_val(model_lst):
    """Evaluate network on given dataset."""

    train_loader, val_loader, test_loader = get_trainloader()
    all_preds=[]
    val_acc = []
    nll_loss=[]
    for model in model_lst:
        indi_acc, indi_loss,max_prob,clean_correct  = evaluate(model, device, val_loader)
        val_acc.append(indi_acc)
        nll_loss.append(indi_loss)
        current_preds,target=evaluate_ensemble(model, device, val_loader)
        all_preds.append(current_preds)
        
    output_mean = torch.mean(torch.stack(all_preds, dim=0), dim=0)
    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    dic={}
    dic['ensemble_accuracy']=correct/float(n)
    dic['individual_accuracy']=val_acc
    dic['individual_loss']=nll_loss
    dic['ensemble_loss']=F.nll_loss(torch.log(output_mean), target, reduction='mean').item()
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_clean_val_accuracy.pkl')
    f=open(path,"wb")
    pickle.dump(dic,f)
    f.close()
    path2=os.path.join(args.save,"max_probs.npy")
    path3=os.path.join(args.save,"corrects.npy")
    np.save(path2,max_prob)
    np.save(path3,clean_correct)



def test_c(model_lst, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    c_accuracy={} 
    max_probs=[]
    c_corrects=[] 
    
    if args.dataset == 'tiny_imgnet':
        mean=[0.480, 0.448, 0.397]
        std=[0.276, 0.269, 0.282]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        for c in CORRUPTIONS:
            for s in range(1, 6):
                valdir = os.path.join(base_path, c, str(s))
                val_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(valdir, test_transform),
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True)

                all_preds=[]
                val_acc = []
                nll_loss=[]
                for model in model_lst:
                    indi_acc, indi_loss, max_prob, c_correct = evaluate(model, device, val_loader)
                    val_acc.append(indi_acc)
                    nll_loss.append(indi_loss)
                    current_preds,target=evaluate_ensemble(model, device, val_loader)
                    all_preds.append(current_preds)
                max_probs.append(max_prob)
                c_corrects.append(c_correct)
                output_mean = torch.mean(torch.stack(all_preds, dim=0), dim=0)
                pred = output_mean.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                n = target.shape[0]
                dic={}
                dic['ensemble_accuracy'] = correct/float(n)
                dic['individual_accuracy'] = val_acc
            
                dic['individual_loss']=nll_loss       
                dic['ensemble_loss']=F.nll_loss(torch.log(output_mean), target, reduction='mean').item()
                c_accuracy[c+'_'+str(s)]=dic
    
    else:
        for corruption in CORRUPTIONS:
            test_data.data = np.load(base_path + corruption + '.npy')
            test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
            test_loader = torch.utils.data.DataLoader(
                                                test_data,
                                                batch_size=args.eval_batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
            all_preds=[]
            val_acc = []
            nll_loss=[]
            for model in model_lst:
                indi_acc, indi_loss,max_prob,c_correct = evaluate(model, device, test_loader)
                val_acc.append(indi_acc)
                nll_loss.append(indi_loss)
                current_preds,target=evaluate_ensemble(model, device, test_loader)
                all_preds.append(current_preds)
            max_probs.append(max_prob)
            c_corrects.append(c_correct)
            output_mean = torch.mean(torch.stack(all_preds, dim=0), dim=0)
            pred = output_mean.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            n = target.shape[0]
            dic={}
            dic['ensemble_accuracy'] = correct/float(n)
            dic['individual_accuracy'] = val_acc
            dic['individual_loss'] =nll_loss       
            dic['ensemble_loss'] =F.nll_loss(torch.log(output_mean), target, reduction='mean').item()
            c_accuracy[corruption] = dic
        
        
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_corruption_accuracy.pkl')
    f=open(path,"wb")
    pickle.dump(c_accuracy,f)
    path2=os.path.join(args.save,"c_max_probs.npy")
    path3=os.path.join(args.save,"c_corrects.npy")
    np.save(path2,max_probs)
    np.save(path3,c_corrects)
    f.close()

def c_test_ensemble_disagreement(model_lst,test_data,base_path):
    disagreement=[]
    for corruption in CORRUPTIONS:
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        test_loader = torch.utils.data.DataLoader(
                                test_data,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True)
        all_preds=[]
        for model in model_lst:
            current_preds,target=evaluate_ensemble(model, device, test_loader)
            all_preds.append(current_preds)
        num_models = len(all_preds)
        dis_matrix = np.zeros(shape=(num_models,num_models))
        predictions = []  
        for i in range(num_models):
            pred_labels = np.argmax(all_preds[i].cpu().numpy(), axis=1)
            predictions.append(pred_labels)
        for i in range(num_models):
            preds1 = predictions[i]
            for j in range(i, num_models):
                preds2 = predictions[j]
                dissimilarity_score = 1 - np.sum(np.equal(preds1, preds2)) / (preds1.shape[0])
                dis_matrix[i][j] = dissimilarity_score
                if i is not j:
                    dis_matrix[j][i] = dissimilarity_score
        disagreement.append(dis_matrix)
        print(f"{corruption} disagreement done")
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_corruption_disagreement.pkl')
    f=open(path,"wb")
    pickle.dump(disagreement,f)
    f.close()

            
    
def test_ensemble_disagreement(model_lst):
    train_loader, val_loader, test_loader = get_trainloader()
    all_preds=[]
    for model in model_lst:
        current_preds,target=evaluate_ensemble(model, device, test_loader)
        all_preds.append(current_preds)
    num_models = len(all_preds)
    dis_matrix = np.zeros(shape=(num_models,num_models))
    predictions = []  
    for i in range(num_models):
        pred_labels = np.argmax(all_preds[i].cpu().numpy(), axis=1)
        predictions.append(pred_labels)

    for i in range(num_models):
        preds1 = predictions[i]
        for j in range(i, num_models):
            preds2 = predictions[j]
            dissimilarity_score = 1 - np.sum(np.equal(preds1, preds2)) / (preds1.shape[0])
            dis_matrix[i][j] = dissimilarity_score
            if i is not j:
                dis_matrix[j][i] = dissimilarity_score
            print(f"dis_{i}_{j}={dis_matrix[i][j]}")
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_clean_disagreement.pkl')
    f=open(path,"wb")
    pickle.dump(dis_matrix,f)
    f.close()
    print("disagreement done")
    
    
def test_ensemble_cka_similarity(model_lst):
    model_pair_cka_lst = []
    for i in range(0, len(model_lst)):
        for j in range(i, len(model_lst)):
            if i != j:
                model1, model2 = model_lst[i], model_lst[j]
                train_loader, val_loader, test_loader = get_trainloader()
                cka_from_features_average = []
                for _ in range(args.CKA_repeat_runs):
                    cka_from_features = []
                    latent_all_1, latent_all_2 = all_latent(model1, model2, train_loader, num_batches = args.CKA_batches, args=args)
                    for name in latent_all_1.keys():
                        
                        if args.flattenHW:
                            cka_from_features.append(feature_space_linear_cka(latent_all_1[name], latent_all_2[name]))
                        else:
                            cka_from_features.append(cka_compute(gram_linear(latent_all_1[name]), gram_linear(latent_all_2[name])))
                        
                    cka_from_features_average.append(cka_from_features)
                cka_from_features_average = np.mean(np.array(cka_from_features_average), axis=0)
                print('cka_from_features shape: ', cka_from_features_average.shape)

                model_pair_cka_lst.append(cka_from_features_average)
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_clean_cka.pkl')
    f=open(path,"wb")
    pickle.dump(model_pair_cka_lst,f)
    f.close()
                
                
def test_ensemble_disagreement_val(model_lst):
    train_loader, val_loader, test_loader = get_trainloader()
    all_preds=[]
    for model in model_lst:
        current_preds,target=evaluate_ensemble(model, device, val_loader)
        all_preds.append(current_preds)
    num_models = len(all_preds)
    dis_matrix = np.zeros(shape=(num_models,num_models))
    predictions = []  
    for i in range(num_models):
        pred_labels = np.argmax(all_preds[i].cpu().numpy(), axis=1)
        predictions.append(pred_labels)

    for i in range(num_models):
        preds1 = predictions[i]
        for j in range(i, num_models):
            preds2 = predictions[j]
            dissimilarity_score = 1 - np.sum(np.equal(preds1, preds2)) / (preds1.shape[0])
            dis_matrix[i][j] = dissimilarity_score
            if i is not j:
                dis_matrix[j][i] = dissimilarity_score
            print(f"dis_{i}_{j}={dis_matrix[i][j]}")
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_clean_val_disagreement.pkl')
    f=open(path,"wb")
    pickle.dump(dis_matrix,f)
    f.close()
    print("disagreement done")


def test_ensemble_KL(model_lst):
    train_loader, val_loader, test_loader = get_trainloader()
    all_preds=[]
    for model in model_lst:
        current_fold_preds, data = evaluate_ensemble_KD( model, test_loader)
        all_preds.append(current_fold_preds)
    num_models = len(all_preds)
    KL_matrix=np.zeros(shape=(num_models,num_models))
    for i in range(num_models):
        preds1 = all_preds[i]
        for j in range(0, num_models):
            preds2 = all_preds[j]
            KL_matrix[i][j]=loss_fn_kd(preds1,preds2).cpu().data.numpy()
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_clean_KL.pkl')
    f=open(path,"wb")
    pickle.dump(KL_matrix,f)
    f.close()
    print("KL done")

def c_test_ensemble_KL(model_lst,test_data,base_path):
    KL=[]
    for corruption in CORRUPTIONS:
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        test_loader = torch.utils.data.DataLoader(
                                test_data,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True)
        all_preds=[]
        for model in model_lst:
            current_preds,target=evaluate_ensemble_KD(model, test_loader)
            all_preds.append(current_preds)
        num_models = len(all_preds)
        KL_matrix=np.zeros(shape=(num_models,num_models))
        for i in range(num_models):
            preds1 = all_preds[i]
            for j in range(0, num_models):
                preds2 = all_preds[j]
                KL_matrix[i][j]=loss_fn_kd(preds1,preds2).cpu().data.numpy()
        KL.append(KL_matrix)
        print(f"{corruption} KL done")
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_corruption_KL.pkl')
    f=open(path,"wb")
    pickle.dump(KL,f)
    f.close()


def test_train(model_lst):
    
    train_loader, _, _ = get_trainloader()
    all_preds=[]
    val_acc = []
    nll_loss=[]
    for model in model_lst:
        indi_acc, indi_loss,max_prob,clean_correct  = evaluate(model, device, train_loader)
        val_acc.append(indi_acc)
        nll_loss.append(indi_loss)
        
    dic={}
    dic['individual_accuracy']=val_acc
    dic['individual_loss']=nll_loss
    #f=open(args.resume+'clean_accuracy.pkl',"wb")
    Path(args.save).mkdir(parents=True, exist_ok=True)
    path=os.path.join(args.save, f'ensemble_m{len(args.resume)}_WA{args.averaging_weight}_noAug_{args.no_aug}_clean_train_accuracy.pkl')
    f=open(path,"wb")
    pickle.dump(dic,f)
    f.close()
    path2=os.path.join(args.save,"max_probs.npy")
    path3=os.path.join(args.save,"corrects.npy")
def extract_prediction(val_loader, model):
    """
    Run evaluation
    """
    model.eval()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            pred = F.softmax(output, dim=1)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    print('* prediction shape = ', y_pred.shape)
    print('* ground truth shape = ', y_true.shape)

    return y_pred, y_true





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
        val_data = torch.utils.data.Subset(datasets.CIFAR10(
                    os.path.join(args.data_path, 'cifar10'), train=True, transform=test_transform, download=True),indices=val_idx)
        test_data = datasets.CIFAR10(
                    os.path.join(args.data_path, 'cifar10'), train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = torch.utils.data.Subset(datasets.CIFAR100(
                    os.path.join(args.data_path, 'cifar100'), train=True, transform=test_transform, download=True),indices=train_idx)
        val_data = torch.utils.data.Subset(datasets.CIFAR100(
                    os.path.join(args.data_path, 'cifar100'), train=True, transform=test_transform, download=True),indices=val_idx)
        test_data = datasets.CIFAR100(
                        os.path.join(args.data_path, 'cifar100'), train=False, transform=test_transform, download=True)
        num_classes = 100
    else:
        train_data = datasets.ImageFolder(args.data_path+'/tiny-imagenet-200/train', test_transform)
        val_data = datasets.ImageFolder(args.data_path+'/tiny-imagenet-200/val', test_transform)
        test_data = datasets.ImageFolder(args.data_path+'/tiny-imagenet-200/val', test_transform)
        num_classes = 200

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    
    return train_loader, val_loader, test_loader

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
            if args.model == 'densenet':
                net = densenet(num_classes=num_classes)
            elif 'wrn-28-10' in args.model:
                print(args.layers,args.width)
                net = WideResNet(28,10 , num_classes, args.droprate)
            elif args.model=='resnet18':
                if args.dataset == 'tiny_imgnet':
                    net = resnet18(args.width,num_classes=num_classes)
                else:
                    net = ResNet18(width=args.width,num_classes=num_classes)
            elif args.model=='resnet34':
                net=ResNet34(width=args.width,num_classes=num_classes)
            elif args.model=='resnet50':
                net=ResNet50(width=args.width,num_classes=num_classes)
            elif args.model=='resnet101':
                net=ResNet101(width=args.width,num_classes=num_classes)
            checkpoint = torch.load(file,map_location='cuda:0')
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            net_ensemble_lst.append(net)
            net = torch.nn.DataParallel(net).cuda()
            print(f'Model restored from: {file}')
            
    
    if args.evaluate:
        if('train' in args.mode):
            test_train(net_ensemble_lst)
        if('predict' in args.mode):
            test(net_ensemble_lst) 
            test_c(net_ensemble_lst, test_data, base_c_path)
        if('disagreement' in args.mode):
            test_ensemble_disagreement_val(net_ensemble_lst)
            test_ensemble_disagreement(net_ensemble_lst)
            c_test_ensemble_disagreement(net_ensemble_lst, test_data, base_c_path)
        if('KL' in args.mode):
            test_ensemble_KL(net_ensemble_lst)
            c_test_ensemble_KL(net_ensemble_lst,test_data,base_c_path)

if __name__ == '__main__':
    main()