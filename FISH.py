import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import os
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from dataset_fine import dataset
import random

from model.alexnet import AlexNet
from model.resnet18 import ResNet18
from config import getConfig
from train_model import train_model


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


if __name__ == '__main__':
    config = getConfig()
    print(config)

    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids

    set_seed(config.random_seed)

    #---------------------------- dataset ------------------------------------
    if config.dataset == 'cub_bird':
        classes = 200
        data_dir = './datasets/cub_bird/'
    elif config.dataset == 'stanford_dog':
        classes = 120
        data_dir = './datasets/stanford_dog/'
    elif config.dataset == 'aircraft':
        classes = 100
        data_dir = './datasets/aircraft/'
    elif config.dataset == 'vegfru':
        classes = 292
        data_dir = './datasets/vegfru/'
    else:
        print('undefined dataset ! ')
    print('Dataset:', config.dataset, ', num of classes:', classes)
    print(data_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),          
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_set = dataset(config.dataset, root_dir=data_dir, transforms = data_transforms['train'], train =True)
    print('train_set',len(train_set))
    val_set = dataset(config.dataset ,root_dir=data_dir, transforms = data_transforms['val'], test =True)
    print('val_set',len(val_set))
    base_set = dataset(config.dataset, root_dir=data_dir, transforms = data_transforms['val'], train =True)
    print('basa_set',len(base_set))

    dataloader = {}
    dataloader['train']= data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    setattr(dataloader['train'], 'total_item_len', len(train_set))
    dataloader['val'] = data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    setattr(dataloader['val'], 'total_item_len', len(val_set))
    dataloader['base'] = data.DataLoader(base_set, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    setattr(dataloader['base'], 'total_item_len', len(base_set))

    # ---------------------------- model ------------------------------------
    if config.model_name == 'resnet18':
        model = ResNet18(config.code_length, classes, config.class_mask)
    elif config.model_name == 'alexnet':
        model = AlexNet(config.code_length, classes, config.class_mask)
    else:
        print('undefined model ! ')
    #print(model)

    model = nn.DataParallel(model)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion_hash = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    save_dir = './training_checkpoint'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(config.model_name+'_'+config.dataset+'_'+str(config.code_length)+'_'+str(config.class_mask)+'.log', 'a')
    log_file.write(str(config))
    print('training start ...')
    train_model(model, dataloader, criterion, criterion_hash, optimizer, exp_lr_scheduler, save_dir, config.epoch, config.code_length, classes, log_file)






