import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.readData import read_dataset
from utils.ResNet import ResNet18
import pandas as pd
import seaborn as sns

# 设置中文显示，使用系统已安装的字体，例如 'SimHei'
plt.rcParams["font.family"] = 'SimHei'
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

# 收集所有预测和真实标签
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 计算整体评估指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"整体准确率: {accuracy:.4f}")
print(f"整体精确率: {precision:.4f}")
print(f"整体召回率: {recall:.4f}")
print(f"整体F1分数: {f1:.4f}")

# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            yticklabels=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 打印分类报告
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

# 转换为DataFrame并显示
df_report = pd.DataFrame(report).transpose()
print("\n分类报告:")
print(df_report.to_string(float_format=lambda x: f"{x:.4f}"))

# 保存为CSV
df_report.to_csv('classification_report.csv')

# 绘制各类别的精确率、召回率和F1分数
metrics_df = df_report.iloc[:-3, :3]  # 排除最后三行(平均值)
plt.figure(figsize=(12, 6))
metrics_df.plot(kind='bar', ax=plt.gca())
plt.title('各类别的精确率、召回率和F1分数')
plt.xlabel('类别')
plt.ylabel('分数')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('class_metrics.png')
plt.show()

# 绘制ROC曲线 (需要二值化标签)
y_test_bin = label_binarize(all_labels, classes=list(range(n_class)))

# 计算每个类别的ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制所有类别的ROC曲线
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], lw=2, 
             label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('各类别的ROC曲线')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.show()

# 绘制PR曲线
from sklearn.metrics import precision_recall_curve, average_precision_score

# 计算每个类别的PR曲线和AP
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_class):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], all_probs[:, i])

# 绘制所有类别的PR曲线
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    plt.plot(recall[i], precision[i], lw=2, 
             label=f'{class_name} (AP = {average_precision[i]:.3f})')

plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('各类别的PR曲线')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('pr_curves.png')
plt.show()

# 保存主要评估指标到CSV
summary_df = pd.DataFrame({
    '指标': ['准确率', '精确率', '召回率', 'F1分数'],
    '数值': [accuracy, precision, recall, f1]
})
summary_df.to_csv('evaluation_summary.csv', index=False)

print("\n评估完成！所有图表已保存到当前目录。")