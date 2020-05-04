import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, mode_3d: bool):
        super().__init__()

        conv = nn.Conv2d if not mode_3d else nn.Conv3d

        modules = []

        if not mode_3d:
            modules.extend([
                torch.nn.utils.spectral_norm(conv(3, 16, 3, 2, 1)),  # 32
                nn.LeakyReLU(0.2, inplace=True),
                # nn.BatchNorm2d(16),

                torch.nn.utils.spectral_norm(conv(16, 32, 3, 2, 1)),  # 16
                nn.LeakyReLU(0.2, inplace=True),
                # nn.BatchNorm2d(32),
            ])
        else:
            modules.extend([
                torch.nn.utils.spectral_norm(conv(3, 32, 3, 2, 1)),  # 16
                nn.LeakyReLU(0.2, inplace=True),
                # nn.BatchNorm2d(32),
            ])

        modules.extend([
            torch.nn.utils.spectral_norm(conv(32, 64, 3, 2, 1)),  # 8
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(64),

            torch.nn.utils.spectral_norm(conv(64, 128, 3, 2, 1)),  # 4
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(128),

            torch.nn.utils.spectral_norm(conv(128, 256, 3, 2, 1)),  # 2
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(256),

            torch.nn.utils.spectral_norm(conv(256, 512, 2, 1, 0)),  # 1
            nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(512),

            torch.nn.utils.spectral_norm(conv(512, 1, 1, 1, 0))
        ])

        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)

        return output
