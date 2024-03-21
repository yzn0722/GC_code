import os

def generate_data_list(data_root, classes_list, mode="train"):
    image_path_list = []
    label_list = []
    if mode == "train":
        for i, classes in enumerate(classes_list):
            image_list = os.listdir(os.path.join(data_root, mode, classes))
            for file in image_list:
                image_path_list.append(os.path.join(data_root, mode, classes, file))
                label_list.append(i)
    else:
        for i, classes in enumerate(classes_list):
            image_list = os.listdir(os.path.join(data_root, mode, classes))
            for file in image_list:
                image_path_list.append(os.path.join(data_root, mode, classes, file))
                label_list.append(i)

    return image_path_list, label_list
