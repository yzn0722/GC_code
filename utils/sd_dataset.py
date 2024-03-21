from torch.utils import data
from PIL import Image

class SdDataset(data.Dataset):
	def __init__(self, img_paths, labels, transform):
		self.imgs = img_paths
		self.labels = labels
		self.transforms = transform

	# 进行切片
	def __getitem__(self, index):
		img = self.imgs[index]
		label = self.labels[index]
		pil_img = Image.open(img)  # pip install pillow
		pil_img = pil_img.convert('RGB')
		data = self.transforms(pil_img)
		return data, label

	# 返回长度
	def __len__(self):
		return len(self.imgs)

