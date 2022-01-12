import os
import numpy as np
import iso_fgan_config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Custom_Dataset(Dataset):

    # root_dir contains all of the image subdirectories
    def __init__(self, root_dir):
        super(Custom_Dataset, self).__init__()
        self.data = []
        # image subdirectories
        self.root_dir = root_dir
        # list of classes
        self.img_dirs = os.listdir(root_dir)

        for idx, name in enumerate(self.img_dirs):
            # retrieve all the images of a single class (e.g. low-res)
            img_path = os.listdir(os.path.join(root_dir, name))
            # add to list of all images, categorised by class
            self.data += list(zip(img_path, [idx] * len(img_path)))

    # __len__ returns the length of the dataset
    def __len__(self):
        return len(self.data)

    # __getitem__ returns a specific image
    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.img_dirs[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = iso_fgan_config.transform(image=image)["image"]

        return image


def test():
    dataset = Custom_Dataset(root_dir=os.path.join(
                                      os.getcwd(),
                                      "images/sims/microtubules/"
    ))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    for low_res, high_res in dataloader:
        print(low_res.shape)
        print(high_res.shape)


# runs the test code without running any imported modules
# (if they contain code)
if __name__ == "__main__":
    test()
