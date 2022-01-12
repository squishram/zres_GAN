import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
from pathlib import Path

# TODO figure out whether the dimensions are the right way round
# pretty sure they're not - fix it!


class Custom_Dataset(Dataset):
    """
    make a dataset for pytorch
    """

    def __init__(self, dir_data, suffix, transform=None):
        """
        go to the directory containing the images
        and extract the relevant files
        for iso_fgan, suffix will be 'lores' or 'hires'
        """

        self.dir_data = Path(dir_data)
        self.files = sorted(self.dir_data.glob('mtubs_sim_*_{}.tif'.format(suffix)))
        self.transform = transform

    def __len__(self):
        """
        denotes the total number of samples
        """

        return len(self.files)

    def __getitem__(self, file):
        """
        obtains one sample of data
        """

        # select image
        img_path = os.path.join(self.dir_data, self.files[file])
        # import image
        img = io.imread(img_path)
        img = np.asarray(img)
        # normalise the pixel values
        img = img - np.mean(img) / np.std(img)
        img = img / np.max(img)
        # transform image if one is supplied
        if self.transform:
            img = self.transform(img)
        img = torch.tensor(img)
        # np.shape(img) = (1, z, y, x):
        img = img.unsqueeze(0)
        return img


def test():
    """
    check that the class outputs datasets are the right size
    """

    # TODO (if you are so inclined) not currently used
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5),
    #                                                      (0.5, 0.5, 0.5)), ])

    # Image filepaths
    lores_filepath = os.path.join(os.getcwd(),
                                  Path("images/sims/microtubules/lores"))
    hires_filepath = os.path.join(os.getcwd(),
                                  Path("images/sims/microtubules/hires"))

    # these are the datasets
    lores_dataset = Custom_Dataset(dir_data=lores_filepath,
                                   suffix="lores")
    hires_dataset = Custom_Dataset(dir_data=hires_filepath,
                                   suffix="hires")

    # these are the dataloaders
    lores_dataloader = DataLoader(lores_dataset,
                                  batch_size=5,
                                  shuffle=True,
                                  num_workers=2)
    hires_dataloader = DataLoader(hires_dataset,
                                  batch_size=5,
                                  shuffle=True,
                                  num_workers=2)

    # create an iterator object from the dataloader to call out batches
    lores_iterator = iter(lores_dataloader)
    hires_iterator = iter(hires_dataloader)

    # pull out a batch!
    # sometimes the iterators 'run out', this stops that from happening
    try:
        lores_batch = next(lores_iterator)
        hires_batch = next(hires_iterator)
    except StopIteration:
        lores_iterator = iter(lores_dataloader)
        hires_iterator = iter(hires_dataloader)
        lores_batch = next(lores_iterator)
        hires_batch = next(hires_iterator)

    # print the batch shape
    print(lores_batch.shape)
    print(hires_batch.shape)


if __name__ == "__main__":
    test()
