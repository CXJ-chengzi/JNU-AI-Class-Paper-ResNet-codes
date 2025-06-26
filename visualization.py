import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.cutout import Cutout
import matplotlib.pyplot as plt


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 0
# 每批加载图数量
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2


def read_dataset(batch_size=batch_size, valid_size=valid_size, num_workers=num_workers, pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_raw = transforms.Compose([
        transforms.ToTensor()
    ])

    # 将数据转换为torch.FloatTensor，并标准化。
    train_data = datasets.CIFAR10(pic_path, train=True,
                                  download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True,
                                  download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False,
                                 download=True, transform=transform_test)

    raw_train_data = datasets.CIFAR10(pic_path, train=True,
                                      download=True, transform=transform_raw)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    raw_train_loader = torch.utils.data.DataLoader(raw_train_data, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, raw_train_loader


def visualize_data(train_loader, raw_train_loader):
    # 获取一个批次的预处理后的数据
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # 获取一个批次的原始数据
    raw_dataiter = iter(raw_train_loader)
    raw_images, _ = next(raw_dataiter)

    # 展示预处理前后的数据对比
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
    for i in range(5):
        # 原始图像
        raw_img = raw_images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(raw_img)
        axes[0, i].axis('off')
        if i == 2:
            axes[0, i].set_title('Raw Images')

        # 预处理后图像
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 2:
            axes[1, i].set_title('Preprocessed Images')
    plt.show()

    # 绘制原始数据分布直方图
    all_raw_images = raw_images.view(-1).numpy()
    plt.figure()
    plt.hist(all_raw_images, bins=50)
    plt.title('Data Distribution of Raw Data')
    plt.xlabel('Pixel Values (Range: [0, 1])')
    plt.ylabel('Frequency')
    plt.show()

    # 绘制预处理后数据分布直方图
    all_images = images.view(-1).numpy()
    plt.figure()
    plt.hist(all_images, bins=50)
    plt.title('Data Distribution of Preprocessed Data')
    plt.xlabel('Pixel Values (Normalized)')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, raw_train_loader = read_dataset()
    visualize_data(train_loader, raw_train_loader)