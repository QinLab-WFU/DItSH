import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import scipy.io as scio
import torch
def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/Loss3_version/FCNET_Triplet/RelaHash-master34444（复件）/Hashcode_DFC15_64_RELA/0.9897945197880331_64-56-DFC15-RELA.mat')
#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/Loss3_version/FCNET_Triplet/RelaHash-master34444（复件）/Hashcode_UCMD_64_RELA/0.9584395022398502_64-52-UCMD-RELA.mat')
#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/multi/ADSH_PyTorch-master/PR_curve/ADSH_DFC1564bits.mat')
#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/multi/ADSH_PyTorch-master/PR_curve/ADSH_UCMD64bits.mat')
#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/multi/HyP2-Loss-main/PR_curve/Hyp2_DFC1564bits.mat')
#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/multi/HyP2-Loss-main/PR_curve/Hyp2_UCMD64bits.mat')
#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/multi/HHF-main/PR_curve/HHF_DFC1564bits.mat')
#data = scio.loadmat('/home/ge/zeroshot/sqp_itriplet/multi/HHF-main/PR_curve/HHF_UCMD64bits.mat')
#data = scio.loadmat('/home/ge/Desktop/retrival/mat/RelaHashDFC15_64bits.mat')
#data = scio.loadmat('/home/ge/Desktop/retrival/mat/RelaHashUCMD64bits.mat')
data = scio.loadmat('/home/ge/Desktop/retrival/mat/SWTH_DFC15_64bits.mat')
#data = scio.loadmat('/home/ge/Desktop/retrival/mat/SWTH_UCMD_64bits.mat')
qb = torch.from_numpy(data['q_img'])
rb = torch.from_numpy(data['r_img'])
ql = torch.from_numpy(data['q_l'])

rl = torch.from_numpy(data['r_l'])
trn_binary = rb.cpu().numpy()
trn_binary = np.asarray(trn_binary, np.int32)
trn_label = rl.cpu().numpy()
tst_binary = qb.cpu().numpy()
tst_binary = np.asarray(tst_binary, np.int32)
tst_label = ql.cpu().numpy()

img_dir = "/home/ge/zeroshot/sqp_itriplet/Loss3_version/FCNET_Triplet/RelaHash-master34444（复件）/data/DFC15 (copy)/"
with open("/home/ge/zeroshot/sqp_itriplet/Loss3_version/FCNET_Triplet/RelaHash-master34444（复件）/data/DFC15 (copy)/database.txt", "r") as f:
    trn_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]
with open("/home/ge/zeroshot/sqp_itriplet/Loss3_version/FCNET_Triplet/RelaHash-master34444（复件）/data/DFC15 (copy)/test.txt", "r") as f:
    tst_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]


m = 1  # query的图像数
n = 10 # 结果返回的图像数
plt.figure(figsize=(40, 20), dpi=50)
font_size = 30
# 随机查询图像
#tst_select_index = np.random.permutation(range(tst_binary.shape[0]))[0: m]
# 固定查询图像
#tst_select_index=[196]
tst_select_index=[i for i in range(1)]

for row, query_index in enumerate(tst_select_index):
    query_binary = tst_binary[query_index]
    query_label = tst_label[query_index]
    # 计算测试集和检索是否相似
    gnd = (np.dot(query_label, trn_label.transpose()) > 0).astype(np.float32)

    # 通过哈希码计算汉明距离
    hamm = CalcHammingDist(query_binary,trn_binary)
    # 计算最近的n个距离的索引
    ind = np.argsort(hamm)[:n]
    # 返回结果的真值
    t_gnd = gnd[ind]
    # 返回结果的汉明距离
    q_hamm = hamm[ind].astype(int)

    q_img_path = tst_img_path[query_index]
    return_img_list = np.array(trn_img_path)[ind].tolist()

    plt.subplot(m, n + 1, row * (n+1) + 1)

    img = Image.open(q_img_path).convert('RGB').resize((128, 128))
    plt.imshow(img)
    plt.axis('off')
    # plt.text(5, 145, 'query image', size=font_size)

    for index, img_path in enumerate(return_img_list):
        # plt.subplot(1, n + 1, index + 2)
        plt.subplot(m, n + 1, row * (n+1) + index + 2)
        img = Image.open(img_path).convert('RGB').resize((120, 120))
        print(img_path)
        if t_gnd[index]:
            plt.text(60, 145, '√', size=font_size)
            img = ImageOps.expand(img, 4, fill=(0, 0, 255))
        else:
            plt.text(60, 145, '×', size=font_size)
            img = ImageOps.expand(img, 4, fill=(255, 0, 0))
        plt.axis('off')
        plt.imshow(img)
#print(return_img_list)
plt.savefig("demo.png")
plt.show()
