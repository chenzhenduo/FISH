import torch
import torch.nn as nn
import random
from utils import label2onehot, calc_train_codes, calc_map_k

def eval_turn(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch_cnt_val, (inputs, labels, _) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            feature_map, outputs, outputs_codes = model(inputs)

            if batch_cnt_val == 0:
                ground = labels
                pred_out = outputs_codes
            else:
                ground = torch.cat((ground,labels))
                pred_out = torch.cat((pred_out,outputs_codes))

    return ground, pred_out

def train_model(model, dataloader, criterion, criterion_hash, optimizer, scheduler, num_epochs, bits, classes, log_file):

    train_codes = calc_train_codes(dataloader, bits, classes)

    for epoch in range(num_epochs):

        model.train()
        ce_loss = 0.0

        for batch_cnt, (inputs, labels, item) in enumerate(dataloader['train']):

            codes = torch.tensor(train_codes[item, :]).float().cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

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
            hide_imgs = inputs * masks
            _, outputs_hide, _ = model(hide_imgs)
            # ------------------------------------------------------------

            loss_class = criterion(outputs_class, labels)
            loss_class_hide = criterion(outputs_hide, labels)
            loss_codes = criterion_hash(outputs_codes, codes)
            loss = loss_class + loss_codes + loss_class_hide  # 0.1*
            loss.backward()
            optimizer.step()
            ce_loss += loss.item() * inputs.size(0)

        epoch_loss = ce_loss / dataloader['train'].total_item_len
        scheduler.step()

        if (epoch+1)%1 == 0:
            ground_q, code_q = eval_turn(model, dataloader['val'])
            ground_d, code_d = eval_turn(model, dataloader['base'])

            labels_onehot_q = label2onehot(ground_q.cpu(), classes)
            labels_onehot_d = label2onehot(ground_d.cpu(), classes)

            map_1 = calc_map_k(torch.sign(code_q), torch.tensor(train_codes).float().cuda(), labels_onehot_q, labels_onehot_d)

            print('epoch:{}:  loss:{:.4f},  MAP:{:.4f}'.format(epoch+1, epoch_loss, map_1))
            log_file.write('epoch:{}:  loss:{:.4f},  MAP:{:.4f}'.format(epoch+1, epoch_loss, map_1) + '\n')

