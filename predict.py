import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor()
])

model_name = "resnet50"
model = torch.load(f"./checkpoint/{model_name}/{model_name}_best.pth", map_location=torch.device("cuda:1"))

i = 0  # 识别图片计数
# 这里最好新建一个test_data文件随机放一些上面整理好的图片进去
root_path = "./datasets/test"  # 待测试文件夹
names = os.listdir(root_path)

label_list = []
pred_list = []
data_class = ["abnormal", "normal"]  # 按文件索引顺序排列
for (index, classes) in enumerate(data_class):
    file_list = os.listdir(os.path.join(root_path, classes))
    for file in tqdm(file_list):
        image_path = os.path.join(root_path, classes, file)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = transform(image)
        image = torch.reshape(image, (1, 3, 384, 384))  # 修改待预测图片尺寸，需要与训练时一致
        image = image.to(torch.device("cuda:1"))
        model.eval()
        with torch.no_grad():
            output = model(image)
        # print(int(output.argmax(1)))
        # 对结果进行处理，使直接显示出预测的种类
        pred_list.append(int(output.argmax(1)))
        label_list.append(int(index))
        # print("名字：{} 图片预测为：{}".format(os.path.join(root_path, classes, file), data_class[int(output.argmax(1))]))


accuracy = accuracy_score(label_list, pred_list)

# 计算 Precision
precision = precision_score(label_list, pred_list, average='macro')

# 计算 Recall
recall = recall_score(label_list, pred_list, average='macro')

# 计算 F1 Score
f1 = f1_score(label_list, pred_list, average='macro')

# 计算Sp
sp = 2 * (precision * recall) / (precision + recall)

print(f"{model_name} Accuracy:", accuracy)
print(f"{model_name} Precision:", precision)
print(f"{model_name} Recall:", recall)
print(f"{model_name} F1 Score:", f1)
print(f"{model_name} SP Score:", sp)

label_mapping = {
    0: "abnormal",
    1: "normal"
}

label_multi = [label_mapping[label] for label in label_list]
pred_mutil = [label_mapping[label] for label in pred_list]

conf_matrix_multi = confusion_matrix(label_multi, pred_mutil)

# 绘制混淆矩阵图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_multi, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(label_multi), yticklabels=np.unique(label_multi))
plt.title(f'{model_name} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks()
plt.yticks()
plt.savefig(f'{model_name}_confusion_matrix.png')

# 初始化变量来存储每个类别的假正例率(fpr)和真正例率(tpr)
label_list = np.array(label_list)
pred_list = np.array(pred_list)

# 计算每个类别的ROC曲线和AUC值
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(label_mapping)):
    fpr[i], tpr[i], _ = roc_curve((label_list == i).astype(int), (pred_list == i).astype(int))
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均ROC曲线和AUC值
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(label_mapping))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(label_mapping)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(label_mapping)
fpr["micro"] = all_fpr
tpr["micro"] = mean_tpr
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 绘制ROC曲线
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue', 'purple', 'brown']
for i, color in zip(range(len(label_mapping)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='{0} (area = {1:0.2f})'
             ''.format(label_mapping[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{model_name} ROC')
plt.legend(loc="lower right")
plt.savefig(f'{model_name}_ROC.png')