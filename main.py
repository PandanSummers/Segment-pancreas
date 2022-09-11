'''
author:zhujunwen
Hubei University of Medicine
'''
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from pickle import TRUE
from random import shuffle
from cv2 import mean
import torch
import argparse
from torch._C import dtype
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset4 import Pandate
from collections import defaultdict
from loss_p import dice_coef
import torch.nn.functional as F
from model.MSAnet import CE_Net_
from lou_loss import iou_score
from model.attention_unet import AttU_Net
from model.unet import Unet
from model.channel_unet import myChannelUnet
from model.FCN import get_fcn8s
from model.MSAnet import MSANET
from model.res34_unet import resnet34_unet
from model.u2net import U2NET
import os
from plot import loss_plot
from plot import metrics_plot
from metrics import get_dice
from metrics import get_iou
import PIL.Image as Image
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])
target_transform=transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/MICCA/train&MICCA", default="train&MICCA")
    parse.add_argument("--epoch", type=int, default=24)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_unet',
                       help='UNet/resnet34_unet/u2net/myChannelUnet/Attention_UNet/Fcn_unet/MSAnet')
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument('--dataset', default='acute pancreatitis',  # dsb2018_256
                       help='Taihe Hospital/aucte pancreatitis')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(1,pretrained=False).to(device)
    if args.arch == 'u2net':
        args.deepsupervision = True
        model = U2NET(3,1).to(device)
    if args.arch =='Attention_UNet':
        model = AttU_Net(3,1).to(device)
    if args.arch == 'myChannelUnet':
        model = myChannelUnet(3,1).to(device)
    if args.arch == 'fcn8s':
        model = get_fcn8s(1).to(device)
    if args.arch == 'MSAnet':
        model = MSANET().to(device)
    return model

def getDataset(args):
    train_dataset = Pandate(r'\traning' transform = trans,target_transform=target_transform)
    train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = Pandate(r'\valing', transform=trans, target_transform=target_transform)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)
    test_dataset = Pandate(r'\MICCA', transform=trans, target_transform=target_transform)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders
def val_model(model,crterion,dataload,best_dice,args):
    model=model.eval()
    print('val')
    with torch.no_grad():
        step=0
        epoch_loss = 0
        epoch_dice=0
        epoch_lou=0
        dt_size = len(dataload.dataset)
        opterizer = optim.Adam(model.parameters())
        for image,label in dataload:
            step+=1
            inputs=image.to(device)
            label=label.to(device)
            opterizer.zero_grad()
            output=model(inputs)
            loss=crterion(output,label)
            dice=dice_coef(output,label)
            lou=iou_score(output,label)
            # print("val,%d/%d,val_loss:%0.4f,dice_loss:%0.3f,Iou_score:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(),dice.item(),lou.item())) 
            epoch_loss+=loss.item()
            epoch_dice+=dice.item()
            epoch_lou+=lou.item()
        print("val,loss:%0.4f dice:%0.3f lou:%0.3f" % (epoch_loss/step,epoch_dice/step,epoch_lou/step))
        mean_loss=epoch_loss/step
        mean_dice=epoch_dice/step
        mean_iou=epoch_lou/step
        if mean_dice>best_dice:
            print('mean_dice:{}>best_dice:{}'.format(mean_dice,best_dice))
            print('=======>save best model!')
            best_dice = mean_dice
            torch.save(model.state_dict(),r'\attt_model_pth'+'best_model'+'{}.pth'.format(args.arch))
        return best_dice,mean_dice,mean_loss
def train(model,crterion, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_dice,aver_hd = 0,0,0,0
    num_epochs = args.epoch
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        opterizer = optim.Adam(model.parameters())
        for image,label in train_dataloader:
            step+=1
            inputs=image.to(device)
            label=label.to(device)
            opterizer.zero_grad()
            output=model(inputs)
            loss=crterion(output,label)
            dice=dice_coef(output,label)
            lou=iou_score(output,label)
            # print("train,%d/%d,train_loss:%0.4f,dice_loss:%0.3f,Iou_score:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(),dice.item(),lou.item()))
            loss.backward() 
            opterizer.step()  
            epoch_loss+=loss.item()
            epoch_dice+=dice.item()
            epoch_lou+=lou.item()  
        print("epoch %d loss:%0.4f dice:%0.3f lou:%0.3f" % (epoch, epoch_loss/step,epoch_dice/step,epoch_lou/step))
        logging.info("epoch %d loss:%0.4f dice:%0.3f lou:%0.3f" % (epoch, epoch_loss/step,epoch_dice/step,epoch_lou/step))
        torch.save(model.state_dict(), '{}_weights_{}.pth.'.format(args.arch,epoch))
        aver_loss=epoch_loss/step
        aver_dice=epoch_dice/step
        aver_iou=epoch_lou/step
        best_dice,mean_dice,mean_lou=val_model(model,crterion,val_dataloader,best_dice)        
        loss_list.append(aver_loss)
        dice_list.append(aver_dice)
        iou_list.append(aver_iou)
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice',iou_list, dice_list)
    return model
def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        loss_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            iou = get_iou(mask_path[0],predict)
            miou_total += iou  #获取当前预测图的miou，并加到总miou中
            dice = get_dice(mask_path[0],predict)
            dice_total += dice

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            #print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict,cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            #print(mask_path[0])
            if save_predict == True:
                aved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
                plt.savefig(saved_predict)
            #plt.pause(0.01)
            print('iou={},dice={}'.format(iou,dice))
            if i < num:i+=1   #处理验证集下一张图
        #plt.show()
        print('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,dice_total/num))
        #print('M_dice=%f' % (dice_total / num))

if __name__ =="__main__":
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\n========' % \
          (args.arch, args.epoch, args.batch_size))
    print('**************************')
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion=nn.BCEWithLogitsLoss()
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)