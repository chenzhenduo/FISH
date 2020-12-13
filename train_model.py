import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
import time
import os
import torch.utils.data as data
from torch.autograd import Variable
from eval_model import eval_turn
import random
from utils import calc_map_k


from torchvision import utils as vutils

def label2onehot(labels, classes):
    labels_onehot = torch.zeros(labels.size()[0],classes)
    index = labels.view(-1,1)
    labels_onehot.scatter_(dim=1, index = index, value = 1)
    #print(labels_onehot.size())
    return labels_onehot

def calc_train_codes(dataloader, bits, classes):
    for batch_cnt, (inputs, labels, item) in enumerate(dataloader['base']):
        if batch_cnt == 0:
            train_labels = labels
        else:
            train_labels = torch.cat((train_labels, labels))
    L_tr = label2onehot(train_labels.cpu(), classes).numpy()

    train_size = L_tr.shape[0]
    sigma = 1
    delta = 0.0001
    myiter = 15

    V = np.random.randn(bits, train_size)
    B = np.sign(np.random.randn(bits, train_size))
    S1, E, S2 = np.linalg.svd(np.dot(B, V.T))
    R = np.dot(S1, S2)
    L = L_tr.T

    for it in range(myiter):

        B = -1 * np.ones((bits, train_size))
        B[(np.dot(R, V)) >= 0] = 1

        Ul = np.dot(sigma * np.dot(L, V.T), np.linalg.pinv(sigma * np.dot(V, V.T)))

        V = np.dot(np.linalg.pinv(sigma * np.dot(Ul.T, Ul) + delta * np.dot(R.T, R)),
                   sigma * np.dot(Ul.T, L) + delta * np.dot(R.T, B))

        S1, E, S2 = np.linalg.svd(np.dot(B, V.T))
        R = np.dot(S1, S2)

    B1 = B.T
    B1 = np.sign(B1)
    print('Code generated!', 'size:', B1.shape)
    return B1


def train_model(model, dataloader, criterion, criterion_hash, optimizer, scheduler, save_dir, num_epochs, bits, classes, log_file):

    train_codes = calc_train_codes(dataloader, bits, classes)

    for epoch in range(num_epochs):

        model.train()

        ce_loss = 0.0
        tic = time.time()

        for batch_cnt, (inputs, labels, item) in enumerate(dataloader['train']):

            codes = torch.tensor(train_codes[item, :]).float().cuda()
            inputs = inputs.cuda()
            labels = torch.from_numpy(np.array(labels)).cuda()

            optimizer.zero_grad()
            feature_map, outputs_class, outputs_codes = model(inputs)

            # ------------------------------------------------------------
            attention = torch.sum(feature_map.detach(), dim=1, keepdim=True)
            attention = nn.functional.interpolate(attention, size=(224, 224), mode='bilinear', align_corners=True)
            masks = []
            for i in range(labels.size()[0]):
                threshold = random.uniform(0.9, 1.0)
                mask = (attention[i] < threshold * attention[i].max()).float()
                masks.append(mask)

            masks = torch.stack(masks)
            drop_imgs = inputs * masks
            _, outputs_drop, outputs_drop_codes = model(drop_imgs)
            # ------------------------------------------------------------

            loss_class = criterion(outputs_class, labels)
            loss_class_drop = criterion(outputs_drop, labels)
            loss_codes = criterion_hash(outputs_codes, codes)
            loss = loss_class + loss_codes + loss_class_drop  # 0.1*
            loss.backward()

            optimizer.step()
            ce_loss += loss.item() * inputs.size(0)

        toc = time.time()
        epoch_loss = ce_loss / dataloader['train'].total_item_len

        print('epoch: {:d} loss: {:f} min: {:4.2f} '.format(epoch, epoch_loss, (toc - tic)), flush=True)
        scheduler.step()
        #save_path = os.path.join(save_dir, 'weights_%d.pth' % 0)  # weights_0.pth
        #torch.save(model.state_dict(), save_path)
        ground_q, code_q = eval_turn(model, dataloader['val'], 'val')
        ground_d, code_d = eval_turn(model, dataloader['base'], 'base')

        print(code_q.size(), code_d.size())
        labels_onehot_q = label2onehot(ground_q.cpu(), classes)
        labels_onehot_d = label2onehot(ground_d.cpu(), classes)

        print(labels_onehot_q.size(), labels_onehot_d.size())
        map_1 = calc_map_k(torch.sign(code_q), torch.tensor(train_codes).float().cuda(), labels_onehot_q,
                           labels_onehot_d)

        print('epoch:', epoch, 'MAP:', map_1)

        log_file.write(str(epoch) + ':\t' + str(epoch_loss) + ':\t' + str(map_1.numpy()) + '\n')
        print('#############################################################')

    log_file.close()
