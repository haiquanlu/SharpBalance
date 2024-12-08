import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from models.wideResnet import WideResNet
from models.resnet import ResNet18, ResNet18_softmax
from models.resnet_imgnet import resnet18
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __getitem__(self, i):
        x, y = self.dataset[i]
        return i,x,y
    def __len__(self):
        return len(self.dataset)

def create_dataset(args, is_fisher=False):

    if args.data=='cifar10':
        mean=[0.4914,0.4822,0.4465]
        std=[0.2470, 0.2435, 0.2616]
    elif args.data=='cifar100':
        mean=[0.5071, 0.4865, 0.4409]
        std=[0.2673, 0.2564, 0.2762]
    else:
        mean=[0.480, 0.448, 0.397]
        std=[0.276, 0.269, 0.282]

    if args.data=='cifar10' or args.data=='cifar100' or args.data=='pets' or args.data=='flowers':
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(size=64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_idx=np.load(args.train_idx_path)
    val_idx= np.load(args.val_idx_path)

    if is_fisher:
        if args.data == 'cifar10':
            train_data = train_data = torch.utils.data.Subset(datasets.CIFAR10(
                args.data_path, train=True, transform=test_transform, download=True),indices=train_idx)
        elif args.data == 'cifar100':
            train_data = train_data = torch.utils.data.Subset(datasets.CIFAR100(
                args.data_path, train=True, transform=test_transform, download=True),indices=train_idx)
        else:
            train_data = datasets.ImageFolder(args.data_path+'/train', test_transform)

        return train_data
    else:
        if args.data == 'cifar10':
            train_data = torch.utils.data.Subset(datasets.CIFAR10(
                        args.data_path, train=True, transform=train_transform, download=True),indices=train_idx)
            val_data = torch.utils.data.Subset(datasets.CIFAR10(
                        args.data_path, train=True, transform=test_transform, download=True),indices=val_idx)
            test_data = datasets.CIFAR10(
                        args.data_path, train=False, transform=test_transform, download=True)
            num_classes= 10
        elif args.data == 'cifar100':
            train_data = torch.utils.data.Subset(datasets.CIFAR100(
                        args.data_path, train=True, transform=train_transform, download=True),indices=train_idx)
            val_data = torch.utils.data.Subset(datasets.CIFAR100(
                        args.data_path, train=True, transform=test_transform, download=True),indices=val_idx)
            test_data = datasets.CIFAR100(
                        args.data_path, train=False, transform=test_transform, download=True)
            
            num_classes = 100

            
        else:
            train_data = datasets.ImageFolder(args.data_path+'/train', train_transform)
            val_data = datasets.ImageFolder(args.data_path+'/val', test_transform)
            test_data = datasets.ImageFolder(args.data_path+'/val', test_transform)
            num_classes = 200
        
        return train_data, val_data, test_data, num_classes

def create_model(args, num_classes, softmax=False):
    # Create model
    if args.model == 'wrn-40-2':
        cls = WideResNet
        cls_args = [40, 2, 10, 0.0]
        cls_args[2] = num_classes
        model = cls(*(cls_args + [args.save_features, args.bench]))
    elif args.model == 'resnet18' and softmax:
        model = ResNet18_softmax(width=args.width,num_classes=num_classes)
    elif args.model == 'resnet18' and args.data == 'tiny_imgnet':
        model = resnet18(args.width,num_classes=num_classes)
    elif args.model == 'resnet18':
        model = ResNet18(width=args.width,num_classes=num_classes)
    
    return model

def save_model(args, model, epoch, best_acc, optimizer, first_stage=False, save_init=False):
    checkpoint = {
        'epoch': epoch,
        'dataset': args.data,
        'model': args.model,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    if not first_stage:
        save_path = os.path.join(args.save_path, f'checkpoint.pth.tar')
    elif save_init:
        save_path = os.path.join(args.save_path, f'checkpoint-{epoch}.pth.tar')
    else:
        save_path = os.path.join(args.save_path, f'checkpoint_startepoch.pth.tar')
    torch.save(checkpoint, save_path)

def load_model(args, model, optimizer):
    assert os.path.isfile(args.resume), "resume should be a file"
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Model restored from epoch:', args.start_epoch)