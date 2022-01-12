import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Custom_Dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for idx, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [idx] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = config.both_transforms(image=image)["image"]
        high_res = config.highres_transform(image=image)["image"]
        low_res = config.lowres_transform(image=image)["image"]
        return low_res, high_res


def test():
    dataset = Custom_Dataset(root_dir=os.path.join(os.getcwd(),
                                                   "images/catface/real/set1/"))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    for low_res, high_res in dataloader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
