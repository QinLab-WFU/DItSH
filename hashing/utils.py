import logging

import numpy as np
import torch

import configs
# import main
from utils.misc import Timer
from save_mat import Save_mat

def get_hamm_dist(codes, centroids, margin=0, normalize=False):
    with torch.no_grad():
        nbit = centroids.size(1)
        dist = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

        if normalize:
            dist = dist / nbit

        if margin == 0:
            return dist
        else:
            codes_clone = codes.clone()
            codes_clone[codes_clone.abs() < margin] = 0
            dist_margin = 0.5 * (nbit - torch.matmul(codes_clone.sign(), centroids.sign().t()))
            if normalize:
                dist_margin = dist_margin / nbit
            return dist, dist_margin


def calculate_mAP(db_codes: object, db_labels: object,
                  test_codes: object, test_labels: object,
                  R: object, ep: object = 0, threshold: object = 0.) -> object:
    # clone in case changing value of the original codes
    print(db_codes.shape,test_codes.shape)
    qb , rb  = db_codes.clone() , test_codes.clone()
    db_codes = db_codes.clone()
    test_codes = test_codes.clone()

    # if value within margin, set to 0
    if threshold != 0:
        db_codes[db_codes.abs() < threshold] = 0
        test_codes[test_codes.abs() < threshold] = 0

    # binarized
    db_codes = torch.sign(db_codes)  # (ndb, nbit)
    test_codes = torch.sign(test_codes)  # (nq, nbit)

    db_labels = db_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    dist = []
    nbit = db_codes.size(1)

    timer = Timer()
    total_timer = Timer()

    timer.tick()
    total_timer.tick()

    with torch.no_grad():
        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        for i, db_code in enumerate(db_codes_loader):
            dist.append(0.5 * (nbit - torch.matmul(test_codes, db_code.t())).cpu())
            timer.toc()
            print(f'Distance [{i + 1}/{len(db_codes_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)  # .numpy()
        print()

    # fast sort
    timer.tick()
    # different sorting will have affect on mAP score! because the order with same hamming distance might be diff.
    # unsorted_ids = np.argpartition(dist, R - 1)[:, :R]

    # torch sorting is quite fast, pytorch ftw!!!
    topk_ids = torch.topk(dist, R, dim=1, largest=False)[1].cpu()
    timer.toc()
    print(f'Sorting ({timer.total:.2f}s)')

    # calculate mAP
    timer.tick()
    APx = []
    for i in range(dist.shape[0]):
        label = test_labels[i, :]
        label[label == 0] = -1
        idx = topk_ids[i, :]
        # idx = idx[np.argsort(dist[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: R], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
        else:  # didn't retrieve anything relevant
            APx.append(0)
        timer.toc()
        print(f'Query [{i + 1}/{dist.shape[0]}] ({timer.total:.2f}s)', end='\r')

    print()

    # def calculate_mAP(db_codes, db_labels,
    #                   test_codes, test_labels,
    #                   R, threshold=0.):

    total_timer.toc()
    l = db_codes.shape[1]
    Save_mat(epoch=ep, output_dim=l, datasets="UCMD", query_labels=test_labels,
             retrieval_labels=db_labels
             , query_img=test_codes, retrieval_img=db_codes, save_dir='.', mode_name="RELA",
             mAP=np.mean(np.array(APx)))
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')


    return np.mean(np.array(APx))


def calculate_accuracy(logits, hamm_dist, labels, multiclass: bool):
    if multiclass:
        pred = logits.topk(5, 1, True, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        # acc = correct[:5].view(-1).float().sum(0, keepdim=True) / logits.size(0)
        acc = correct[:5].reshape(-1).float().sum(0, keepdim=True) / logits.size(0)

        pred = hamm_dist.topk(5, 1, False, True)[1].t()
        correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        # cbacc = correct[:5].view(-1).float().sum(0, keepdim=True) / hamm_dist.size(0)
        cbacc = correct[:5].reshape(-1).float().sum(0, keepdim=True) / hamm_dist.size(0)
    else:
        acc = (logits.argmax(1) == labels.argmax(1)).float().mean()
        cbacc = (hamm_dist.argmin(1) == labels.argmax(1)).float().mean()

    return acc, cbacc
