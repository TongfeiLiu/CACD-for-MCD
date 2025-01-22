from model.net import CAE, COAE
from utils.loss import JS_loss, l2_distance
import argparse
from utils.data import Data_Loader, to_normalization
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import cv2
import os
from utils.Evaluation import Evaluation
from skimage.filters.thresholding import threshold_otsu
from sklearn import metrics
import scipy.io as io

# bastrop dataset:      nc1 = 11, nc2 = 3
# california dataset:   nc1 = 7, nc2 = 10
# shuguang dataset:     nc1 = 1, nc2 = 3
# italy dataset:        nc1 = 1, nc2 = 3
# france dataset:       nc1 = 3, nc2 = 3
# yellow dataset:       nc1 = 1, nc2 = 1
# gloucester2 dataset:  nc1 = 3, nc2 = 1
# gloucester1 dataset:  nc1 = 1, nc2 = 3

# 选择设备，有cuda用cuda，没有就用cpu
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='yellow', type=str)  # bastrop or california or other
# parser.add_argument('--t1_path', default='./data/France/Img7-Ac.png', type=str)
# parser.add_argument('--t2_path', default='./data/France/Img7-Bc.png', type=str)
# parser.add_argument('--gt_path', default='./data/France/Img7-C.png', type=str)
# parser.add_argument('--t1_path', default='./data/California/California.mat', type=str)
# parser.add_argument('--t1_path', default='./data/Bastrop/Cross-sensor-Bastrop-data.mat', type=str)
# parser.add_argument('--t1_path', default='./data/Italy/Italy_1.bmp', type=str)
# parser.add_argument('--t2_path', default='./data/Italy/Italy_2.bmp', type=str)
# parser.add_argument('--gt_path', default='./data/Italy/Italy_gt.bmp', type=str)
parser.add_argument('--t1_path', default='./data/Yellow/yellow_C_1.bmp', type=str)
parser.add_argument('--t2_path', default='./data/Yellow/yellow_C_2.bmp', type=str)
parser.add_argument('--gt_path', default='./data/Yellow/gt.png', type=str)
# parser.add_argument('--t1_path', default='./data/Shuguang/shuguang_1.bmp', type=str)
# parser.add_argument('--t2_path', default='./data/Shuguang/shuguang_2.bmp', type=str)
# parser.add_argument('--gt_path', default='./data/Shuguang/shuguang_gt.bmp', type=str)
# parser.add_argument('--t1_path', default='./data/Gloucester1/Img5-A.png', type=str)
# parser.add_argument('--t2_path', default='./data/Gloucester1/Img5-Bc.png', type=str)
# parser.add_argument('--gt_path', default='./data/Gloucester1/Img5-C.png', type=str)
# parser.add_argument('--t1_path', default='./data/Gloucester2/T1-Img17-Bc.png', type=str)
# parser.add_argument('--t2_path', default='./data/Gloucester2/T2-Img17-A.png', type=str)
# parser.add_argument('--gt_path', default='./data/Gloucester2/Img17-C.png', type=str)

parser.add_argument('--t1_nc', default='1', type=int)
parser.add_argument('--t2_nc', default='1', type=int)
parser.add_argument('--patch_size', default=3, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_delay', default=range(1, 30, 1), type=float)
parser.add_argument('--optim', default='rmsprop', type=str)  # rmsprop
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--COAE_eps', default=10, type=int)
parser.add_argument('--vision_path', default='./vision/', type=str)
args = parser.parse_args()

CAENet = CAE(in_channels=3, patch_size=args.patch_size)
COAENet_t1 = COAE()
COAENet_t2 = COAE()


CAENet.to(device=device)
COAENet_t1.to(device=device)
COAENet_t2.to(device=device)

trans = Transforms.Compose([Transforms.ToTensor()])

train_dataset = Data_Loader(data_name=args.data_name, t1_path=args.t1_path,
                            t2_path=args.t2_path, gt_path=args.gt_path,
                            patch_size=args.patch_size, mode='test',
                            transform=Transforms.ToTensor())

test_dataset = Data_Loader(data_name=args.data_name, t1_path=args.t1_path,
                           t2_path=args.t2_path, gt_path=args.gt_path,
                           patch_size=args.patch_size, mode='test',
                           transform=Transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size * 6,
                                          shuffle=False)

if args.data_name == 'bastrop':
    mat = io.loadmat(args.t1_path)
    x1 = mat['t1_L5'][:, :, 3] # landsat-5: NIR band
    x2 = mat["t2_ALI"][:, :, 5] # OE-AIL: NIR band
    x1 = to_normalization(x1)
    x2 = to_normalization(x2)
    x1 = x1[..., np.newaxis]
    x2 = x2[..., np.newaxis]
    gt2 = mat["ROI_1"]
    gt = gt2 * 255
    o_h, o_w = gt.shape[0], gt.shape[1]
    # cv2.imwrite(args.vision_path + "gt.png", gt2 * 255)

    cv2.imwrite(args.vision_path + "t1.png", x1*255)
    cv2.imwrite(args.vision_path + "t2.png", x2*255)
elif args.data_name == 'california_mat':
    mat = io.loadmat(args.t1_path)
    x1 = mat['image_t1'][:, :, 0] # SAR
    x2 = mat["image_t2"][:, :, 3]+1 # landsat-8: NIR band
    x1 = to_normalization(x1)
    # x2 = to_normalization(x2)
    x1 = x1[..., np.newaxis]
    x2 = x2[..., np.newaxis]
    gt2 = mat["gt"]
    gt = gt2 * 255
    o_h, o_w = gt.shape[0], gt.shape[1]
    # cv2.imwrite(args.vision_path + "gt.png", gt2 * 255)
    cv2.imwrite(args.vision_path + "t1.png", x1*255)
    cv2.imwrite(args.vision_path + "t2.png", x2*255)
elif args.data_name == 'california':
    x1 = cv2.imread(args.t1_path)
    x2 = cv2.imread(args.t2_path)
    gt = cv2.imread(args.gt_path)[:, :, 0] # 0-255
    o_h, o_w = gt.shape[0], gt.shape[1]
    gt2 = gt / 255 # 0-1

    cv2.imwrite(args.vision_path + "image_t1.png", x1)
    cv2.imwrite(args.vision_path + "image_t2.png", x2)
else:
    x1 = cv2.imread(args.t1_path)
    x2 = cv2.imread(args.t2_path)
    gt = cv2.imread(args.gt_path)[:, :, 0]  # 0-255
    o_h, o_w = gt.shape[0], gt.shape[1]
    gt2 = gt / 255  # 0-1

    cv2.imwrite(args.vision_path + "t1.png", x1)
    cv2.imwrite(args.vision_path + "t2.png", x2)

ps = args.patch_size
t1_expand = cv2.copyMakeBorder(x1, ps // 2, ps // 2, ps // 2, ps // 2, cv2.BORDER_CONSTANT, 0)  # cv2.BORDER_DEFAULT
t2_expand = cv2.copyMakeBorder(x2, ps // 2, ps // 2, ps // 2, ps // 2, cv2.BORDER_CONSTANT, 0)
h, w = t1_expand.shape[0], t1_expand.shape[1]

def train():
    # BCE_loss = nn.BCELoss().cuda()
    # MSE_loss = nn.MSELoss(reduction='mean').cuda()

    # Starting training CAE
    opt_CAENet = optim.Adam(CAENet.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler_CAENet = torch.optim.lr_scheduler.MultiStepLR(opt_CAENet, milestones=args.lr_delay, gamma=0.999)

    for epoch in range(args.epochs):
        with tqdm(total=len(train_loader), desc='CAE Train Epoch #{}'.format(epoch + 1), ncols=190) as t:
            for batch_idx, (t1, t2, _) in tqdm(enumerate(train_loader)):
                CAENet.train()
                COAENet_t1.eval()
                COAENet_t2.eval()

                opt_CAENet.zero_grad()

                t1 = t1.to(device=device).float()
                t2 = t2.to(device=device).float()

                _, t1_hat = CAENet(t1)
                _, t2_hat = CAENet(t2)
                loss_const_t1 = torch.sum(torch.abs((t1 - t1_hat)))
                loss_const_t2 = torch.sum(torch.abs((t2 - t2_hat)))

                loss = loss_const_t1 + loss_const_t2

                loss.backward()
                opt_CAENet.step()

                t.set_postfix({'lr': '%.6f' % opt_CAENet.param_groups[0]['lr'],
                               'loss': '%.5f' % (loss.item()),
                               'Const_t1': '%.5f' % (loss_const_t1.item()),
                               'Const_t2': '%.5f' % (loss_const_t2.item())
                               })
                t.update(1)

        # scheduler_CAENet.step()
        print('\n')

    # Starting training COAE
    opt_COAENet_t1 = optim.Adam(COAENet_t1.parameters(), lr=args.lr, weight_decay=1e-5)
    opt_COAENet_t2 = optim.Adam(COAENet_t2.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler_COAENet_t1 = torch.optim.lr_scheduler.MultiStepLR(opt_COAENet_t1, milestones=args.lr_delay, gamma=0.999)
    scheduler_COAENet_t2 = torch.optim.lr_scheduler.MultiStepLR(opt_COAENet_t2, milestones=args.lr_delay, gamma=0.999)

    for epoch in range(args.COAE_eps):
        with tqdm(total=len(train_loader), desc='COAE Train Epoch #{}'.format(epoch + 1), ncols=190) as t:
            for batch_idx, (t1, t2, _) in tqdm(enumerate(train_loader)):
                CAENet.eval()
                COAENet_t1.train()
                COAENet_t2.train()
                opt_COAENet_t1.zero_grad()
                opt_COAENet_t2.zero_grad()

                t1 = t1.to(device=device).float()
                t2 = t2.to(device=device).float()

                t1_feature, _ = CAENet(t1)
                t2_feature, _ = CAENet(t2)

                t1_f = COAENet_t1(t1_feature)
                t2_f = COAENet_t2(t2_feature)

                t1_const = torch.sum(torch.abs((t1_feature - t1_f)))
                D1 = torch.sum(torch.abs((t1_f - t2_feature)))
                loss1 = t1_const + D1

                t2_const = torch.sum(torch.abs((t2_feature - t2_f)))
                D2 = torch.sum(torch.abs((t2_f - t1_feature)))
                loss2 = t2_const + D2

                loss1.backward(retain_graph=True) # retain_graph=True
                opt_COAENet_t1.step()

                loss2.backward()
                opt_COAENet_t2.step()

                t.set_postfix({'lr': '%.6f' % opt_COAENet_t1.param_groups[0]['lr'],
                               'loss1': '%.5f' % (loss1.item()),
                               'loss2': '%.5f' % (loss2.item()),
                               'D1': '%.5f' % (D1.item()),
                               'D2': '%.5f' % (D2.item())
                               })
                t.update(1)

        scheduler_COAENet_t1.step()
        scheduler_COAENet_t2.step()

        acc = test(args.patch_size, epoch)
        print("\n")
        if acc:
            if not os.path.exists(args.vision_path + str(epoch + 1)):
                os.makedirs(args.vision_path + str(epoch + 1))
            torch.save(CAENet.state_dict(),
                       args.vision_path + str(epoch + 1) + '/CAE_ps_' + str(args.patch_size) + '_epoch_' + str(
                           args.epochs+1) + '.pth')
            torch.save(COAENet_t1.state_dict(),
                   args.vision_path + str(epoch + 1) + '/COAENet_t1_ps_' + str(args.patch_size) + '_epoch_' + str(
                       epoch+1) + '.pth')
            torch.save(COAENet_t2.state_dict(),
                   args.vision_path + str(epoch + 1) + '/COAENet_t2_ps_' + str(args.patch_size) + '_epoch_' + str(
                       epoch+1) + '.pth')

def test(patch_size, epoch):
    CAENet.eval()
    COAENet_t1.eval()
    COAENet_t2.eval()

    res1 = []
    res2 = []
    Gres = []

    with tqdm(total=len(test_loader), desc='Test Epoch #{}'.format(epoch + 1), ncols=170, colour='cyan') as t:
        for batch_idx, (t1, t2, _) in tqdm(enumerate(test_loader)):
            bat = t1.shape[0]
            t1 = t1.to(device=device).float()
            t2 = t2.to(device=device).float()

            t1_feature, _ = CAENet(t1)
            t2_feature, _ = CAENet(t2)

            t1_f = COAENet_t1(t1_feature)
            t2_f = COAENet_t2(t2_feature)

            for i in range(bat):
                D1 = torch.sum(torch.abs((t1_f[i] - t2_feature[i])))
                D2 = torch.sum(torch.abs((t2_f[i] - t1_feature[i])))
                diff = D1 + D2
                diff = diff.detach().cpu().numpy()
                Gres.append(diff)
            t.update(1)

    Gmin = np.min(np.array(Gres))
    Gmax = np.max(np.array(Gres))
    Gchageres = (np.array(Gres) - Gmin) / (Gmax - Gmin)

    Gchageres = Gchageres.reshape(o_h, o_w)

    FPR_3, TPR_3, thres_3 = metrics.roc_curve(gt2.flatten(), Gchageres.flatten())

    AUC3 = metrics.auc(FPR_3, TPR_3)

    Gchageres = Gchageres * 255

    # segmentation
    thre3 = threshold_otsu(Gchageres)

    CM3 = (Gchageres > thre3) * 255

    # gt = cv2.imread(args.gt_path)[:, :, 0]
    # gt = (gt > 150) * 255

    Indicators3 = Evaluation(gt, CM3)
    OA3, kappa3, AA3 = Indicators3.Classification_indicators()
    P3, R3, F13 = Indicators3.ObjectExtract_indicators()
    TP3, TN3, FP3, FN3 = Indicators3.matrix()

    print('AUC={}, OA={}, kappa={}'.format(AUC3 ,OA3, kappa3))

    if OA3 >= 60 or kappa3 >= 20.0: # To select a better result, set it yourself, and delete it if you don’t need it.
        path = args.vision_path + '{}'.format(epoch + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        val_acc = open(args.vision_path + str(epoch + 1) + '/val_acc.txt', 'a')
        val_acc.write(
            '===============================Parameters settings==============================\n')
        val_acc.write('=== epoch={} || train ps={} ===\n'.format((epoch+1), patch_size))
        val_acc.write('============================================================================\n')
        val_acc.write('TP={} || TN={} || FP={} || FN={}\n'.format(TP3, TN3, FP3, FN3))
        val_acc.write("\"AUC\":\"" + "{}\"\n".format(AUC3))
        val_acc.write("\"OA\":\"" + "{}\"\n".format(OA3))
        val_acc.write("\"Kappa\":\"" + "{}\"\n".format(kappa3))
        val_acc.write("\"AA\":\"" + "{}\"\n".format(AA3))
        val_acc.write("\"Precision\":\"" + "{}\"\n".format(P3))
        val_acc.write("\"Recall\":\"" + "{}\"\n".format(R3))
        val_acc.write("\"F1\":\"" + "{}\"\n".format(F13))
        val_acc.close()

        cv2.imwrite(args.vision_path + str(epoch + 1) + '/Diff_' + 'ps_' + str(patch_size) + '_' + str(epoch + 1) + '.png', Gchageres)
        ################################################################################################################
        cv2.imwrite(args.vision_path + str(epoch + 1) + '/CM_' + 'ps_' + str(patch_size) + '_' + str(epoch + 1) + '.png', CM3)

        return True
    else:
        return False


if __name__ == "__main__":
    train()
    # train_BCOAE()
    # test_BCOAE()
    # test()
