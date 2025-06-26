import torch
import torch.nn as nn
from utils.readData import read_dataset
from utils.ResNet import ResNet18
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 100

# 加载数据集
_, _, test_loader = read_dataset(batch_size=batch_size, pic_path='dataset')

# 初始化模型并加载预训练权重
model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class)
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)
model.eval()

# CIFAR-10类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 反归一化函数
def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将图像从归一化状态恢复到原始状态"""
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image

# --------------------------
# 可视化样本预测结果
# --------------------------
def visualize_predictions(num_samples=10):
    """可视化随机样本的预测结果"""
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, 1)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 4))
    
    for i in range(num_samples):
        # 左侧：显示图像及预测结果
        img = denormalize(images[i])
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        
        # 设置标题颜色 - 如果预测正确为绿色，错误为红色
        title_color = 'green' if predictions[i].item() == labels[i].item() else 'red'
        title = f"真实: {classes[labels[i]]} | 预测: {classes[predictions[i]]} | 置信度: {confidences[i]:.2%}"
        axes[i, 0].set_title(title, color=title_color, fontsize=12)
        
        # 右侧：显示各类别的预测概率分布
        ax = axes[i, 1]
        sns.barplot(x=classes, y=probs[i].cpu().numpy(), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("各类别预测概率分布", fontsize=10)
        ax.set_ylabel("概率")
        ax.tick_params(axis='x', rotation=90)
        
        # 标记正确类别和预测类别
        ax.axvline(x=labels[i].item(), color='blue', linestyle='--', alpha=0.5, label='真实类别')
        ax.axvline(x=predictions[i].item(), color='orange', linestyle='--', alpha=0.5, label='预测类别')
        ax.legend()
        
        # 在柱状图上方显示具体概率值
        for j, prob in enumerate(probs[i].cpu().numpy()):
            ax.text(j, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300)
    plt.show()

# --------------------------
# 可视化最容易混淆的类别
# --------------------------
def visualize_most_confused_classes(top_n=5):
    """可视化最容易混淆的类别对"""
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    # 计算每个类别被错误分类的次数
    misclassifications = {}
    for i in range(n_class):
        for j in range(n_class):
            if i != j:
                misclassifications[(i, j)] = cm[i, j]
    
    # 按错误次数排序
    sorted_misclassifications = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)
    
    # 可视化最容易混淆的类别对
    fig, axes = plt.subplots(top_n, 3, figsize=(15, top_n * 5))
    
    for i, ((true_idx, pred_idx), count) in enumerate(sorted_misclassifications[:top_n]):
        # 找到被错误分类的样本
        misclassified_indices = np.where((np.array(all_labels) == true_idx) & 
                                        (np.array(all_preds) == pred_idx))[0]
        
        if len(misclassified_indices) > 0:
            # 随机选择一个样本
            sample_idx = np.random.choice(misclassified_indices)
            
            # 获取对应的图像
            dataiter = iter(test_loader)
            batch_idx = sample_idx // batch_size
            for _ in range(batch_idx + 1):
                images, labels = next(dataiter)
            
            sample_image_idx = sample_idx % batch_size
            image = images[sample_image_idx].to(device)
            
            # 预测
            with torch.no_grad():
                output = model(image.unsqueeze(0))
                prob = torch.nn.functional.softmax(output, dim=1)[0]
            
            # 显示原始图像
            img = denormalize(image)
            axes[i, 0].imshow(img)
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f"真实: {classes[true_idx]}\n预测: {classes[pred_idx]}\n错误次数: {count}", fontsize=10)
            
            # 显示真实类别的典型样本
            true_class_samples = np.where(np.array(all_labels) == true_idx)[0]
            true_sample_idx = np.random.choice(true_class_samples)
            
            batch_idx = true_sample_idx // batch_size
            dataiter = iter(test_loader)
            for _ in range(batch_idx + 1):
                images, _ = next(dataiter)
            
            true_image_idx = true_sample_idx % batch_size
            true_image = images[true_image_idx].to(device)
            
            axes[i, 1].imshow(denormalize(true_image))
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f"{classes[true_idx]}的典型样本", fontsize=10)
            
            # 显示预测类别的典型样本
            pred_class_samples = np.where(np.array(all_labels) == pred_idx)[0]
            pred_sample_idx = np.random.choice(pred_class_samples)
            
            batch_idx = pred_sample_idx // batch_size
            dataiter = iter(test_loader)
            for _ in range(batch_idx + 1):
                images, _ = next(dataiter)
            
            pred_image_idx = pred_sample_idx % batch_size
            pred_image = images[pred_image_idx].to(device)
            
            axes[i, 2].imshow(denormalize(pred_image))
            axes[i, 2].axis('off')
            axes[i, 2].set_title(f"{classes[pred_idx]}的典型样本", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('confused_classes.png', dpi=300)
    plt.show()

# --------------------------
# 可视化高置信度错误样本
# --------------------------
def visualize_high_confidence_errors(num_samples=5):
    """可视化高置信度但预测错误的样本"""
    high_confidence_errors = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)
            
            # 找出预测错误的样本
            for i in range(len(labels)):
                if predictions[i] != labels[i]:
                    high_confidence_errors.append(
                        (inputs[i], labels[i], predictions[i], confidences[i], probs[i])
                    )
    
    # 按置信度排序
    high_confidence_errors.sort(key=lambda x: x[3], reverse=True)
    
    # 可视化最高置信度的错误样本
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 4))
    
    for i in range(min(num_samples, len(high_confidence_errors))):
        image, true_label, pred_label, confidence, prob = high_confidence_errors[i]
        
        # 显示图像
        img = denormalize(image)
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"真实: {classes[true_label]}\n预测: {classes[pred_label]}\n置信度: {confidence:.2%}", fontsize=10)
        
        # 显示概率分布
        ax = axes[i, 1]
        sns.barplot(x=classes, y=prob.cpu().numpy(), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("各类别预测概率分布", fontsize=10)
        ax.set_ylabel("概率")
        ax.tick_params(axis='x', rotation=90)
        
        # 标记正确类别和预测类别
        ax.axvline(x=true_label.item(), color='blue', linestyle='--', alpha=0.5, label='真实类别')
        ax.axvline(x=pred_label.item(), color='orange', linestyle='--', alpha=0.5, label='预测类别')
        ax.legend()
        
        # 在柱状图上方显示具体概率值
        for j, p in enumerate(prob.cpu().numpy()):
            ax.text(j, p + 0.02, f'{p:.2f}', ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('high_confidence_errors.png', dpi=300)
    plt.show()

# --------------------------
# 可视化低置信度正确样本
# --------------------------
def visualize_low_confidence_corrects(num_samples=5):
    """可视化低置信度但预测正确的样本"""
    low_confidence_corrects = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)
            
            # 找出预测正确但置信度低的样本
            for i in range(len(labels)):
                if predictions[i] == labels[i] and confidences[i] < 0.7:  # 只考虑置信度低于0.7的样本
                    low_confidence_corrects.append(
                        (inputs[i], labels[i], predictions[i], confidences[i], probs[i])
                    )
    
    # 按置信度排序
    low_confidence_corrects.sort(key=lambda x: x[3])
    
    # 可视化最低置信度的正确样本
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 4))
    
    for i in range(min(num_samples, len(low_confidence_corrects))):
        image, true_label, pred_label, confidence, prob = low_confidence_corrects[i]
        
        # 显示图像
        img = denormalize(image)
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"真实: {classes[true_label]}\n预测: {classes[pred_label]}\n置信度: {confidence:.2%}", fontsize=10)
        
        # 显示概率分布
        ax = axes[i, 1]
        sns.barplot(x=classes, y=prob.cpu().numpy(), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("各类别预测概率分布", fontsize=10)
        ax.set_ylabel("概率")
        ax.tick_params(axis='x', rotation=90)
        
        # 标记正确类别和预测类别
        ax.axvline(x=true_label.item(), color='blue', linestyle='--', alpha=0.5, label='真实类别')
        ax.axvline(x=pred_label.item(), color='orange', linestyle='--', alpha=0.5, label='预测类别')
        ax.legend()
        
        # 在柱状图上方显示具体概率值
        for j, p in enumerate(prob.cpu().numpy()):
            ax.text(j, p + 0.02, f'{p:.2f}', ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('low_confidence_corrects.png', dpi=300)
    plt.show()

# 运行所有可视化函数
visualize_predictions()
visualize_most_confused_classes()
visualize_high_confidence_errors()
visualize_low_confidence_corrects()