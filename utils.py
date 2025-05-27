from os import listdir, path

import torch


def get_dir_files(dir, *, split=''):
    """Return all files in all directories in the given directory."""
    files = []
    for directory in listdir(dir):
        if split and split not in directory:
            continue

        for filename in listdir(path.join(dir, directory)):
            files.append(path.join(directory, filename))
    return sorted(files)


def spherical_to_cartesian(spherical):
    """Convert spherical tensor to cartesian coordinates."""
    azimuth_rad = torch.deg2rad(spherical[..., 0])
    elevation_rad = torch.deg2rad(spherical[..., 1])
    x = torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = torch.sin(elevation_rad)
    return x, y, z


def cartesian_to_spherical(cartesian):
    """Convert cartesian tensor to spherical coordinates."""
    x, y, z = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
    radius = torch.sqrt(x**2 + y**2 + z**2)
    azimuth_rad = torch.atan2(y, x)
    elevation_rad = torch.asin(z / radius)
    azimuth_deg = torch.rad2deg(azimuth_rad)
    elevation_deg = torch.rad2deg(elevation_rad)
    return azimuth_deg, elevation_deg, radius
