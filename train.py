import os

import timm
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from utils import generate_data_list, SdDataset

train_transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor()
])

data_root = "./datasets"
classes_list = [
    "abnormal", "normal"
]

model_name = "resnet50"
BATCH_SIZE = 64

# 读取训练数据路径，验证数据路径
train_image_path_list, train_label_list = generate_data_list(data_root, classes_list, mode="train")
val_image_path_list, val_label_list = generate_data_list(data_root, classes_list, mode="test")

train_dataset = SdDataset(train_image_path_list, train_label_list, train_transform)
val_dataset = SdDataset(val_image_path_list, val_label_list, val_transform)

train_dl = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)  # 训练集标签
test_dl = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)  # 测试集标签

# 判断是否使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "CUDA:0")
# model_ft = Swin_T(num_classes=2)
# model_ft = timm.create_model(model_name="mobilenetv3_large_100", num_classes=8)
model_ft = timm.create_model(model_name=model_name, num_classes=2)

model = model_ft.to(DEVICE)  # 将模型迁移到gpu

# 优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(DEVICE)  # 将loss_fn迁移到GPU
# Adam损失函数
optimizer = optim.Adam(model.parameters(), lr=5e-5)
writer = SummaryWriter(log_dir=os.path.join('checkpoint', model_name, 'tf_logs'))


def train(epoch):
    model.train()
    correct = 0
    total = 0
    train_acc = 0
    train_loss = 0
    total_loss = 0
    # 开始迭代每个batch中的数据
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        # 计算损失
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # 计算准确率
        train_acc = correct / total
        train_loss = total_loss / total
        # 每训练100个batch打印一次训练集的loss和准确率
        if (batch_idx + 1) % 5 == 0:
            print('Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                  batch_idx + 1,
                                                                                  train_loss,
                                                                                  train_acc))

    # 计算每个epoch内训练集的acc
    total_train_acc.append(train_acc), total_train_loss.append(train_loss)
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Accuracy', train_acc, epoch)


# Testing
def test(epoch, ckpt):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_acc = 0
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            test_acc = correct / total
            test_loss = total_loss / total
        print(
            'Epoch-{}-Test Accurancy: {:.3f}'.format(epoch + 1, test_acc), '\n')

    total_test_acc.append(test_acc), total_test_loss.append(test_loss)
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', test_acc, epoch)

    # 保存权重文件
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir(os.path.join('checkpoint', model_name)):
            os.mkdir(os.path.join('checkpoint', model_name))
        torch.save(model, os.path.join('checkpoint', model_name, model_name + "_" + ckpt))
        best_acc = acc


total_test_acc = []
total_train_acc = []
total_test_loss = []
total_train_loss = []
# 开始训练
epoch = 50
best_acc = 0
for epoch in range(epoch):
    train(epoch)
    test(epoch, "best.pth")
    # 动态调整学习率
    optimizer.step()

writer.close()
plt.figure()
plt.plot(range(epoch + 1), total_train_acc, label='Train Accurancy')
plt.plot(range(epoch + 1), total_test_acc, label='Test Accurancy')
plt.xlabel('Epoch')
plt.ylabel('Accurancy')
plt.title('Accurancy')
plt.legend()
plt.savefig(os.path.join('checkpoint', model_name, f'{model_name}_Accurancy.jpg'))  # 自动保存plot出来的图片

plt.figure()
plt.plot(range(epoch + 1), total_train_loss, label='Train Loss')
plt.plot(range(epoch + 1), total_test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig(os.path.join('checkpoint', model_name, f'{model_name}_Loss.jpg'))  # 自动保存plot出来的图片

# 输出best_acc
print(f'Best Acc: {best_acc}%')
