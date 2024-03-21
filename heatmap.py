import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# 加载模型
model_name = "MobileNetV3"
model_path = f"./checkpoint/{model_name}/{model_name}_best.pth"
model = torch.load(model_path, map_location=torch.device("cuda:1"))
model.eval()

# 图像转换
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # 你需要根据你的模型输入大小调整大小
    transforms.ToTensor()
])

# 加载图像
image_path = "/data2/zhangzifan/code_dir/2024-02-08-01/datasets_cls/test/adenosis/SOB_B_A-14-22549AB-40-023.png"
image = Image.open(image_path).convert('RGB')

# 前向传播
with torch.no_grad():
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # 添加 batch 维度
    image = image.to(torch.device("cuda:1"))
    activations = []  # 用于存储每一层的输出特征图

    # 遍历模型的每一层
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hook = module.register_forward_hook(
                lambda module, input, output: activations.append(output.cpu().numpy())
            )

    model(image)  # 执行前向传播

    # 创建文件夹来保存特征图
    output_dir = "feature_maps"
    os.makedirs(output_dir, exist_ok=True)

    # 将特征图保存到文件夹中
    for i, activation in enumerate(activations):
        layer_output_dir = os.path.join(output_dir, f"layer_{i}")
        os.makedirs(layer_output_dir, exist_ok=True)
        for j in range(activation.shape[1]):  # 通道数
            feature_map = activation[0, j, :, :]
            feature_map_image = Image.fromarray((feature_map * 255).astype('uint8'))
            feature_map_image.save(os.path.join(layer_output_dir, f"channel_{j}_feature_map.png"))
