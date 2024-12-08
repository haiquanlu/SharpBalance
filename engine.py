import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import make_functional_with_buffers, vmap, grad
import numpy as np
from tqdm import tqdm

from utils import create_dataset, CustomDataset

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def train_one_epoch(args, net, train_loader, optimizer, normal_indice, scheduler, epoch, device):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    criterion = nn.CrossEntropyLoss().to(device)
    for i, (indice, images, targets) in enumerate(train_loader):
      images, targets = images.to(device), targets.to(device)
      optimizer.zero_grad()
      if(args.sam=='False'):
        logits = net(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
      else:
        _,loss=mix_step(args, net,indice,images,targets,normal_indice,optimizer,epoch,device)
        
      loss_ema +=loss

      if i % args.print_freq == 0:
        min_lr = 100
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
        print('Iter {:d}/{:d}, Train Loss {:.3f}, lr {:.5f}'.format(i, len(train_loader), loss_ema, min_lr))

    return loss_ema

def mix_step(args, net,indice,data,target,idx,optimizer,epoch, device):
    criterion = nn.CrossEntropyLoss().to(device)
    flat_indice=[]
    normal_indice=[]
    correct=0
    total_loss=0
    for i in range(data.shape[0]):
        if(indice[i].data.item() in idx):
            normal_indice.append(i)
        else:
            flat_indice.append(i)
    flat_target=target[flat_indice]
    flat_data=data[flat_indice]
    flat_ratio=len(flat_indice)/len(indice)
    if(len(normal_indice)>0):
        #print("has normal indice")
        normal_data=data[normal_indice]
        normal_target=target[normal_indice]
        normal_ratio=len(normal_indice)/len(indice)
        output=net(normal_data)
        loss=criterion(output,normal_target)
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(normal_target.view_as(pred)).sum().item()
        
        loss=loss*normal_ratio
        total_loss+=loss
        loss.backward()
        optimizer.normal_step(zero_grad=True)  
    if(len(flat_indice)>0):
        #print("has flat data")
        output=net(flat_data)
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(flat_target.view_as(pred)).sum().item()
        loss=criterion(output,flat_target)
        total_loss+=loss*flat_ratio
        loss.backward()
        optimizer.first_step(zero_grad=True)

        output=net(flat_data)
        loss=criterion(output,flat_target)
        loss=loss*flat_ratio
        loss.backward()
        optimizer.second_step(zero_grad=False)
    else:
        optimizer.third_step(zero_grad=False)
    return correct,total_loss

def evaluate(args, net, test_loader,device):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.to(device), targets.to(device)
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def evaluate_c(args, net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  corruption_accs_dict = {}
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = evaluate(net, test_loader)
    corruption_accs.append(test_acc)
    corruption_accs_dict[corruption] = test_acc
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs), corruption_accs_dict


# sharpbalance
def sam_idx_complemetary(args, flats, i, ascend=True):
    print(i)
    indice=np.arange(flats[0].shape[0])
    if(args.flat_ratio!=0):
        total=int(indice.shape[0]*args.flat_ratio)
        for j in flats.keys():
            print(len(flats.keys()))
            flat=flats[j]
            indice1=np.argsort(flat)
            if(ascend):
                indice=np.intersect1d(indice,indice1[:total])
            else:
                indice=np.intersect1d(indice,indice1[-total:])
        this_indice=indice[i::len(flats.keys())]
        return this_indice
    else:
        return np.array([-1])

def boosting_sam_split(args, flats, i, ascend=False):
    normal_indice = np.arange(flats[0].shape[0])
    sam_indice = np.array([])
    total=int(flats[0].shape[0]*args.flat_ratio)
    for j in flats.keys():
        if j != i:
            indice_model = np.argsort(flats[j])
            if ascend:
                indice_model = indice_model[:total]
            else:
                indice_model = indice_model[-total:]
            sam_indice = np.union1d(sam_indice, indice_model)
    # others
    sharp_indice = np.array([])
    total=int(flats[0].shape[0]*args.flat_ratio)
    for j in flats.keys():
        indice_model = np.argsort(flats[j])
        if ascend:
            indice_model = indice_model[:total]
        else:
            indice_model = indice_model[-total:]
        sharp_indice = np.union1d(sharp_indice, indice_model)
    flat_indice = list(set(normal_indice) - set(sharp_indice))
    flat_split = flat_indice[i::len(flats.keys())]
    
    sharp_indice = list(np.union1d(sam_indice, flat_split))
    normal_indice = list(set(normal_indice) - set(sharp_indice))
    return normal_indice

def boosting_sam_split_boost(args, flats, i, ascend=False):
    normal_indice = np.arange(flats[0].shape[0])
    sam_indice = np.array([])
    total=int(flats[0].shape[0]*args.flat_ratio)
    for j in flats.keys():
        if j != i:
            indice_model = np.argsort(flats[j])
            if ascend:
                indice_model = indice_model[:total]
            else:
                indice_model = indice_model[-total:]
            sam_indice = np.union1d(sam_indice, indice_model)
    # others
    sharp_indice = np.array([])
    total=int(flats[0].shape[0]*args.flat_ratio)
    for j in flats.keys():
        indice_model = np.argsort(flats[j])
        if ascend:
            indice_model = indice_model[:total]
        else:
            indice_model = indice_model[-total:]
        sharp_indice = np.union1d(sharp_indice, indice_model)
    flat_indice = list(set(normal_indice) - set(sharp_indice))
    flat_split = flat_indice[i::len(flats.keys())]
    sam_indice_boost = list(set(flat_indice) - set(flat_split))
    
    sharp_indice = np.union1d(sam_indice, sam_indice_boost)
    normal_indice = list(set(normal_indice) - set(sharp_indice))
    return normal_indice

def random_subset(args, flats, i):
    normal_indice_num = int(flats[0].shape[0] * (1 - args.flat_ratio))
    normal_indice = np.array(np.random.choice(np.arange(flats[0].shape[0]), size=normal_indice_num, replace=False))
    return normal_indice
    
def same_subset(args, flats, i):
    rng = np.random.RandomState(0)
    normal_indice_num = int(flats[0].shape[0] * (1 - args.flat_ratio))
    normal_indice = np.array(rng.choice(np.arange(flats[0].shape[0]), size=normal_indice_num, replace=False))
    return normal_indice


def compute_loss_stateless_model(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    loss_fn=nn.CrossEntropyLoss()
    global fmodel
    predictions = fmodel(params, buffers, batch) 
    loss = loss_fn(predictions, targets)
    return loss

def test_fisher(net, train_loader, flat_trial, device):
    net.eval()
    flatness=torch.zeros(len(train_loader.dataset))
    global fmodel
    fmodel, params, buffers = make_functional_with_buffers(net)
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    for trial in range(flat_trial):
        for i, dt in enumerate(tqdm(train_loader)):
            idx,data,target=dt
            data,target=data.to(device, dtype=torch.float), target.to(device)             
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, target)
            ft_per_sample_grads=list(ft_per_sample_grads)
            for j in range(len(ft_per_sample_grads)):
                #print(ft_per_sample_grads[j].shape)
                ft_per_sample_grads[j]=ft_per_sample_grads[j].view(data.shape[0],-1)            
            grads=torch.cat(ft_per_sample_grads,dim=1)
            fisher=grads.pow(2).sum(dim=1)
            flatness[idx]+=fisher.cpu().detach()
    flatness=flatness/flat_trial
    return flatness.numpy()

def test_sharpness_loss(model, train_loader, optimizer, criterion, device):
    sharpness = torch.zeros(len(train_loader.dataset))
    for i, dt in enumerate(tqdm(train_loader)):
        idx, batch, targets= dt
        batch, targets = batch.to(device, dtype=torch.float), targets.to(device)
        optimizer.zero_grad()

        outputs = model(batch)
        loss_erm = criterion(outputs, targets)
        loss_erm_scalar = torch.mean(loss_erm)
        loss_erm_scalar.backward()
        optimizer.first_step(zero_grad=True)

        outputs = model(batch)
        loss_sam = criterion(outputs, targets)
        optimizer.back_step()

        loss_sharpness = loss_sam - loss_erm
        sharpness[idx] += loss_sharpness.cpu().detach()
    
    return sharpness



def main_fisher(args,process_id,dict_to_share,model):
    use_cuda = torch.cuda.is_available()
    count = torch.cuda.device_count()
    cuda_id = process_id % count
    device = torch.device(f"cuda:{cuda_id}" if use_cuda else "cpu")
    print(f'device will be chosen as {device} for this run.')

    # Load datasets
    train_data = create_dataset(args, is_fisher=True)
    train_data=CustomDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    dict_to_share[process_id]=test_fisher(model,train_loader,args.flat_trial,device)
    return dict_to_share[process_id]

def main_sharpness_loss(args, process_id, dict_to_share, model, optimizer):
    use_cuda = torch.cuda.is_available()
    count = torch.cuda.device_count()
    cuda_id = process_id % count
    device = torch.device(f"cuda:{cuda_id}" if use_cuda else "cpu")
    print(f'device will be chosen as {device} for this run.')

    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    # Load datasets
    train_data = create_dataset(args, is_fisher=True)
    train_data=CustomDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    dict_to_share[process_id] = test_sharpness_loss(model, train_loader, optimizer, criterion, device)

    return dict_to_share[process_id]


   