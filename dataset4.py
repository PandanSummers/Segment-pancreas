from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms
import glob
import torch
import os 
from torch.utils.data import Dataset, DataLoader
def read_file(datapath):
    file1=[]
    for id in os.listdir(datapath):
        file_1=os.path.join(datapath,id)
        file_1=file_1+'\\'+'train'
        for file in os.listdir(file_1):
            file_0=os.path.join(file_1,file)
            file1.append(file_0)
    return file1
def sum_images(datapath):
    sum_images_data=[]
    for i in range(len(datapath)):
        for j in datapath[i]:
            sum_images_data.append(j)
    return sum_images_data
class Pandate(Dataset):
    def __init__(self, data_path,transform=None, target_transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        data_path_1=read_file(data_path)
        self.imgs_path=data_path_1
        # print(data_path_1)
        self.transform = transform
        self.target_transform = target_transform
        # 单独的训练集    
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # print(image_path)
        # 根据image_path生成label_path
        label_path = image_path.replace('train', 'trainmask')
        # 读取训练图片和标签图片
        # print(image_path)
        # # print(label_path)
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        # image = Image.open(image_path)
        # label = Image.open(label_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label= self.target_transform(label)
        # 将数据转为单通道的图片
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
   
if __name__ == "__main__":
    train_path = r'F:\dl_pancreas_model\val'
    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
    target_transform=transforms.ToTensor()
    train_set= Pandate(train_path, transform = trans,target_transform=target_transform)
    train_loader=torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=8, 
                                               shuffle=True)