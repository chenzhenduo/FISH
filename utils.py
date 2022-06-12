import torch
import numpy as np


def label2onehot(labels, classes):
    labels_onehot = torch.zeros(labels.size()[0],classes)
    index = labels.view(-1,1)
    labels_onehot.scatter_(dim=1, index = index, value = 1)
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
    print('Code generated,', 'size:', B1.shape)
    return B1

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.int)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd, as_tuple=False)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_L = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_L = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    map = calc_map_k(qB, rB, query_L, retrieval_L)
    print(map)
