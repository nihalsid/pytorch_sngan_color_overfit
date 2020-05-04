import argparse

import os

import torch
import torch.utils.data
from PIL import Image
import numpy as np

from dataset import Dataset
from discriminator import Discriminator
from generator import Generator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def train(opts):
    # Define environment
    set_gpus(opts.gpu)
    device = torch.device("cuda")

    # Other params
    batch_size: int = 32
    latent_dimension: int = 8
    validation_size: int = 36
    noise_shape = (batch_size, latent_dimension, 1, 1, 1) if opts.mode_3d else (batch_size, latent_dimension, 1, 1)

    fake_target = torch.zeros(batch_size, 1, 1, 1, device=device)
    real_target = torch.ones(batch_size, 1, 1, 1, device=device)

    # Define validation params
    z_validation = torch.randn(validation_size, latent_dimension, 1, 1, device=device)

    # fix for dims in case of 3D
    if opts.mode_3d:
        fake_target = torch.zeros(batch_size, 1, 1, 1, 1, device=device)
        real_target = torch.ones(batch_size, 1, 1, 1, 1, device=device)
        z_validation = torch.randn(validation_size, latent_dimension, 1, 1, 1, device=device)

    visualizer = make_and_save_volume_grid if opts.mode_3d else make_and_save_image_grid
    os.makedirs(opts.output_path, exist_ok=True)

    # Define models
    generator = Generator(latent_dimension, mode_3d=opts.mode_3d).to(device, non_blocking=True)
    discriminator = Discriminator(mode_3d=opts.mode_3d).to(device, non_blocking=True)

    # Define train data loader
    max_iterations: int = 200000
    dataset = Dataset(max_iterations * batch_size, mode_3d=opts.mode_3d)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=validation_size, shuffle=False, pin_memory=True)

    # Define optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.99))

    criterion = torch.nn.functional.binary_cross_entropy_with_logits

    # Export some real images
    real_sample_images = to_rgb(next(iter(val_dataloader)))
    visualizer(real_sample_images, os.path.join(opts.output_path, f"real.png"))

    # Train loop
    for iteration, images in enumerate(train_dataloader):
        # Move data to gpu
        images = images.to(device, non_blocking=True)

        # Train generator
        # sample z
        z = torch.randn(noise_shape, device=device)
        # get G(z): pass z through generator --> get prediction
        fake_sample = generator(z)
        # pass G(z) through discriminator
        fake_prediction = discriminator(fake_sample)
        # compute fake loss
        loss_generator = criterion(fake_prediction, real_target)

        # backprop through generator
        optimizer_g.zero_grad()
        loss_generator.backward()
        optimizer_g.step()

        # Train discriminator
        # pass real data through discriminator
        real_prediction = discriminator(images)
        # pass G(z).detach() through discriminator
        fake_prediction = discriminator(fake_sample.detach())

        # compute real loss
        loss_real = criterion(real_prediction, real_target)

        # compute fake loss
        loss_fake = criterion(fake_prediction, fake_target)
        loss_discriminator = (loss_real + loss_fake) / 2

        # backprop through discriminator
        optimizer_d.zero_grad()
        loss_discriminator.backward()
        optimizer_d.step()

        if iteration % opts.log_frequency == opts.log_frequency - 1:
            log_fragments = [
                f"{iteration + 1:>5}:",
                f"Loss(G): {loss_generator.item():>5.4f}",
                f"Loss(D): {loss_discriminator.item():>5.4f}",
                f"Real Pred.: {torch.sigmoid(real_prediction).mean().item():>5.4f}",
                f"Fake Pred.: {torch.sigmoid(fake_prediction).mean().item():>5.4f}",
            ]
            print(*log_fragments, sep="\t")

        # Validation
        if iteration % opts.validation_frequency == opts.validation_frequency - 1:
            with torch.no_grad():
                generator.eval()
                val_samples = generator(z_validation).to("cpu")
                generator.train()

            # output image
            val_grid_path = os.path.join(opts.output_path, f"{iteration + 1:05d}.png")
            visualizer(to_rgb(val_samples), val_grid_path)


def set_gpus(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)


def to_rgb(batch: np.array) -> np.array:
    batch = batch.permute(0, 2, 3, 1).numpy() if len(batch.shape) == 4 else batch.permute(0, 2, 3, 4, 1).numpy()
    batch = (batch + 1) / 2
    batch = (batch * 255).astype(np.uint8)
    return batch


def image_grid(images: np.array, padding: int) -> Image:
    num_images = int(np.sqrt(images.shape[0]))
    sample_width = images.shape[1]
    grid_size = num_images * sample_width + (num_images - 1) * padding
    grid = Image.new("RGB", (grid_size, grid_size))

    for i in range(num_images):
        for j in range(num_images):
            index = i * num_images + j
            sample = Image.fromarray(images[index], mode="RGB")
            pos_x = j * (sample_width + padding)
            pos_y = i * (sample_width + padding)
            grid.paste(sample, (pos_x, pos_y))

    return grid


def make_and_save_image_grid(images: np.array, path: str, padding: int = 5) -> None:
    _image_grid = image_grid(images, padding)
    _image_grid.save(path)


def make_and_save_volume_grid(volume: np.array, path: str) -> None:
    alpha = np.ones(volume.shape[1:4] + (1,), dtype=np.float32)
    skip_factor = 3
    images = []
    for i in range(volume.shape[0]):
        current_volume = volume[i] / 255
        current_volume = np.concatenate((current_volume, alpha), axis=3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = np.meshgrid(list(range(0, current_volume.shape[0], skip_factor)), list(range(0, current_volume.shape[1], skip_factor)), list(range(0, current_volume.shape[2], skip_factor)))
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        ax.scatter(x, y, z, c=current_volume[x, y, z, :].reshape((-1, 4)), marker='o')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        images.append(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
        plt.close()
    make_and_save_image_grid(np.array(images), path, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--output_path", default="out", type=str)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--validation_frequency", type=int, default=100)
    parser.add_argument("--mode_3d", action='store_true')

    args = parser.parse_args()
    train(args)
