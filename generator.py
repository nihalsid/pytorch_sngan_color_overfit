import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dimension: int, mode_3d: bool) -> None:
        super().__init__()

        self.verbose = True
        conv = nn.Conv2d if not mode_3d else nn.Conv3d
        conv_t = nn.ConvTranspose2d if not mode_3d else nn.ConvTranspose3d
        norm = nn.BatchNorm2d if not mode_3d else nn.BatchNorm3d
        modules = [
            conv(latent_dimension, 64, 1, 1, 0, bias=False),  # 1
            # norm(32),
            nn.ReLU(inplace=True),

            conv_t(64, 64, 2, 2, 0, bias=False),  # 2
            norm(64),
            nn.ReLU(inplace=True),

            conv(64, 128, 3, 1, 1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),

            conv_t(128, 128, 4, 2, 1, bias=False),  # 4
            norm(128),
            nn.ReLU(inplace=True),

            conv(128, 128, 3, 1, 1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),

            conv_t(128, 128, 4, 2, 1, bias=False),  # 8
            norm(128),
            nn.ReLU(inplace=True),

            conv(128, 128, 3, 1, 1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),

            conv_t(128, 128, 4, 2, 1, bias=False),  # 16
            norm(128),
            nn.ReLU(inplace=True),

            conv(128, 128, 3, 1, 1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),

            conv_t(128, 128, 4, 2, 1, bias=False),  # 32
            norm(128),
            nn.ReLU(inplace=True),

        ]

        if not mode_3d:
            modules.extend([
                conv(128, 128, 3, 1, 1, bias=False),
                norm(128),
                nn.ReLU(inplace=True),

                conv_t(128, 128, 4, 2, 1, bias=False),  # 64
                norm(128),
                nn.ReLU(inplace=True),

                conv(128, 64, 3, 1, 1, bias=False),
                norm(64),
                nn.ReLU(inplace=True),
            ])
        else:
            modules.extend([
                conv(128, 64, 3, 1, 1, bias=False),
                norm(64),
                nn.ReLU(inplace=True),
            ])

        modules.extend([
            conv(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        ])

        self.model = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z

        for depth, module in enumerate(self.model.children()):
            shape_before = x.size()
            x = module(x)
            shape_after = x.size()
            if self.verbose is True:
                print(f"{depth:02d}: {shape_before} --> {shape_after}")

        self.verbose = False

        output = x

        # output: torch.Tensor = self.model(z)

        return output
