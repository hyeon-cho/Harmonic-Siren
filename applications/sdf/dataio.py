"""Utilities used by the Armadillo SDF experiment.

Adapted from the SIREN implementation; see LICENSE.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}
