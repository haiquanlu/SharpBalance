import argparse
import os
import shutil
import time
import random
import copy
import numpy as np
import torch.multiprocessing as mp
from sam.sam import NSAM
import torch
import torch.backends.cudnn as cudnn

from utils import create_dataset, create_model, CustomDataset, save_model, load_model
from engine import train_one_epoch, evaluate, evaluate_c, main_fisher, sam_idx_complemetary, boosting_sam_split, boosting_sam_split_boost, random_subset, same_subset

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default='cifar10', help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument("--train_idx_path",type=str, default='dataset/train_idx.npy')
parser.add_argument("--val_idx_path",type=str, default='dataset/valid_idx.npy')
parser.add_argument('--model', type=str, default='resnet18', help='Choose architecture.')
parser.add_argument('--width', type=int, default=64)

# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--lrsche',type=str, default='multistep')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# Checkpointing options
parser.add_argument('--save_path', type=str, help='Folder to save checkpoints.')
parser.add_argument('--resume', type=str, default=None, help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument('--print-freq', type=int, default=50, help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument('--num-workers', type=int, default=4, help='Number of pre-fetching threads.')

# for sparse
parser.add_argument('--bench', default=False, action='store_true')
parser.add_argument('--save_features', default=False, action='store_true')

# params for sharpbalance
parser.add_argument('--initial_epochs', type=int, default=-1, help='')
parser.add_argument('--flat_trial', default=1, type=int)
parser.add_argument('--rho',default=0.05,type=float)
parser.add_argument('--flat_ratio',default=0.,type=float)
parser.add_argument('--seed', nargs='+', metavar='S', help='random seed (default: 17)')
parser.add_argument('--current_seed', type=int, default=17, metavar='S')
parser.add_argument('--sam',default='False',type=str)
parser.add_argument('--random', type=str,default='False')

parser.add_argument('--is_resume', default=False, action='store_true')
parser.add_argument('--sharpness_type', default='fisher', type=str)
parser.add_argument('--data_selection_type', default='sharpbalance', type=str)
parser.add_argument('--train_subset', type=str, default='boosting')


args = parser.parse_args()


def main(args, process_id, dict_to_share, barrier, idx_to_share, idx_barrier):
    # ser seed for reproducibility
    np.random.seed(args.current_seed)
    random.seed(args.current_seed)
    torch.manual_seed(args.current_seed)
    torch.cuda.manual_seed(args.current_seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(args.current_seed)
    print(f"Random seed set as {args.current_seed}")
    count=torch.cuda.device_count()
    cuda_id=process_id%count
    device = torch.device(f"cuda:{cuda_id}")

    print(args)
    # Load datasets
    train_data, val_data, test_data, num_classes = create_dataset(args)
    train_data= CustomDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)  

    # model
    model = create_model(args, num_classes)

    # optimizer
    if(args.sam=='True'):
        print("using sam")
        base_optimizer = torch.optim.SGD
        optimizer = NSAM(model.parameters(),base_optimizer,rho=args.rho,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Distribute model across all visible GPUs
    model = model.to(device)
    cudnn.benchmark = True

    args.start_epoch = 0
    best_acc = 0
    if args.is_resume:
        args.resume = os.path.join(args.resume, 'checkpoint_startepoch.pth.tar')
        load_model(args, model=model, optimizer=optimizer)
        print(args.start_epoch)
    if args.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc = evaluate(args, model, val_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(test_loss, 100 - 100. * test_acc))

    # lr scheduler
    if(args.lrsche=='multistep'):
        if(args.sam=='True'):
            if args.is_resume:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer,
				    milestones=[int(args.epochs*3/4)], last_epoch=args.start_epoch-1)
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer,
				    milestones=[int(args.epochs/2), int(args.epochs*3/4)], last_epoch=-1)
            
        else:
            if args.is_resume:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer.base_optimizer,
				    milestones=[int(args.epochs*3/4)], last_epoch=args.start_epoch-1)
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
				    milestones=[int(args.epochs/2), int(args.epochs*3/4)], last_epoch=-1)
    else:
        print('using cosine scheduler\n')
        if(args.sam=='True'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epochs)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # model saveing path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.save_path):
        raise Exception('%s is not a dir' % args.save_path)
    # log path
    log_path = os.path.join(args.save_path, args.data + '_' + args.model + f'_training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')
    
    # training loop

    if(args.sam=='True'):
        normal_indice=np.array([-1])
    else:
        normal_indice=np.arange(len(train_loader.dataset))


    print('Beginning trrging from epoch:', args.start_epoch + 1)
    for epoch in range(args.start_epoch, args.epochs):
        begin_time = time.time()

        # if first stage finished, measuring dataset sharpness
        if (epoch-1)==args.initial_epochs and args.sam=='True' and args.flat_ratio != 0:
            data_sharpness = main_fisher(args, process_id, dict_to_share, model)

            barrier.wait()
            print("complete flatness measure ")
            
            if args.train_subset == 'random':
                normal_indice = random_subset(args, dict_to_share, process_id)
            elif args.train_subset == 'same':
                normal_indice = same_subset(args, dict_to_share, process_id)
            else:
                if args.data == 'cifar10':
                    normal_indice=boosting_sam_split_boost(args, dict_to_share, process_id, ascend=True)
                else:
                    normal_indice=boosting_sam_split(args, dict_to_share, process_id, ascend=True)
            
        train_loss = train_one_epoch(args, model, train_loader, optimizer, normal_indice ,scheduler, epoch, device)

        # evaluation and save best model
        test_loss, test_acc = evaluate(args, model, val_loader, device)
        is_best = test_acc > best_acc
        scheduler.step()
        best_acc = max(test_acc, best_acc)
        save_model(args, model, epoch, best_acc, optimizer)
        if is_best:
            save_path = os.path.join(args.save_path, f'checkpoint.pth.tar')
            shutil.copyfile(save_path, os.path.join(args.save_path, f'model_best.pth.tar'))

        # log
        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % ((epoch + 1), time.time() - begin_time, train_loss, test_loss, 100 - 100. * test_acc))
        print(
            'Epoch {0:3d} | Time {1:5d} s | Train Loss {2:.4f} | Test Loss {3:.3f} |'
            ' Test Error {4:.2f}'
            .format((epoch + 1), int(time.time() - begin_time), train_loss, test_loss, 100 - 100. * test_acc))


if __name__ == '__main__':
    with mp.Manager() as manager:
        dict_to_share = manager.dict()
        idx_to_share=manager.dict()
        barrier=(mp.Barrier(len(args.seed)))
        idx_barrier=mp.Barrier(len(args.seed))
        processes=[]
        print(len(args.seed))
        for i in range(len(args.seed)):
            current_seed=int(args.seed[i])
            copy_arg=copy.deepcopy(args)
            copy_arg.current_seed=current_seed
            copy_arg.save_path= copy_arg.save_path+f'seed_{current_seed}'
            if args.is_resume:
                copy_arg.resume= copy_arg.resume+f'seed_{current_seed}'
            else:
                copy_arg.resume= f'seed_{current_seed}'
            p=mp.Process(target=main,args=(copy_arg,i,dict_to_share, barrier,idx_to_share,idx_barrier))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


    
          
    