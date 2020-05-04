import torch.utils.data
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, max_iterations, mode_3d=False):
        super().__init__()
        self.mode_3d = mode_3d
        self.volume_shape = 32 if mode_3d else 64
        # ColorBrewer Qualitative 8-class Set1 color palette
        self.colors = np.array([
            (228, 26, 28),
            (55, 126, 184),
            (77, 175, 74),
            (152, 78, 163),
            (254, 127, 1),
            (254, 254, 51),
            (166, 86, 40),
            (247, 129, 191)
        ])

        self.max_iterations = max_iterations

    def __getitem__(self, index):
        random_color_index = np.random.randint(0, self.colors.shape[0], size=1).item()
        if self.mode_3d:
            random_color = self.colors[random_color_index][:, np.newaxis, np.newaxis, np.newaxis]
            volume = np.repeat(random_color, self.volume_shape, axis=1)
            volume = np.repeat(volume, self.volume_shape, axis=2)
            volume = np.repeat(volume, self.volume_shape, axis=3)
        else:
            random_color = self.colors[random_color_index][:, np.newaxis, np.newaxis]
            volume = np.repeat(random_color, self.volume_shape, axis=1)
            volume = np.repeat(volume, self.volume_shape, axis=2)
        volume = (volume.astype(np.float32) / 255.0) * 2 - 1
        return volume

    def __len__(self):
        return self.max_iterations
