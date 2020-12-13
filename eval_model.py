from __future__ import print_function, division
import time
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import pdb


def eval_turn(model, data_loader, version):

    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = data_val[0].cuda()
            labels = torch.from_numpy(np.array(data_val[1])).cuda()

            feature_map, outputs, outputs_codes = model(inputs)

            if batch_cnt_val == 0:
                ground = labels
                pred_out = outputs_codes
            else:
                ground = torch.cat((ground,labels))
                pred_out = torch.cat((pred_out,outputs_codes))

        t1 = time.time()
        since = t1-t0

    return ground, pred_out

