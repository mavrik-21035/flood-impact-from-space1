"""GeoTIFF data loading for the local Kuro Siwo segmentation trainer."""

from pathlib import Path

import numpy as np
import rasterio
import torch


class KuroSiwoSegmentationDataset(torch.utils.data.Dataset):
    """Load paired Kuro Siwo Sentinel-1 patches and flood-label rasters."""

    raster_extensions = {".tif", ".tiff"}

    def __init__(
        self,
        root_dir,
        image_prefix="MS1",
        mask_prefix="MK0_MLU",
        valid_prefix="MK0_MNA",
        ignore_index=255,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.ignore_index = int(ignore_index)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder does not exist: {self.root_dir}")

        files_by_folder = {}
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.raster_extensions:
                files_by_folder.setdefault(path.parent, []).append(path)

        self.samples = []
        for folder, paths in sorted(files_by_folder.items()):
            image_path = self._find_raster(paths, image_prefix)
            mask_path = self._find_raster(paths, mask_prefix)
            if image_path is None or mask_path is None:
                continue
            valid_path = self._find_raster(paths, valid_prefix) if valid_prefix else None
            self.samples.append((image_path, mask_path, valid_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No paired {image_prefix} imagery and {mask_prefix} labels found under {self.root_dir}"
            )

    @staticmethod
    def _find_raster(paths, prefix):
        if not prefix:
            return None
        matches = sorted(path for path in paths if path.stem.startswith(prefix))
        if len(matches) > 1:
            raise ValueError(
                f"Expected one {prefix} raster per sample folder, found {len(matches)} in {matches[0].parent}"
            )
        return matches[0] if matches else None

    @staticmethod
    def _read_raster(path):
        with rasterio.open(path) as dataset:
            return dataset.read()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, valid_path = self.samples[index]
        image = self._read_raster(image_path).astype(np.float32, copy=False)
        mask = self._read_raster(mask_path)[0]

        if image.shape[-2:] != mask.shape:
            raise ValueError(
                f"Image and label dimensions differ for {image_path.parent}: "
                f"{image.shape[-2:]} vs {mask.shape}"
            )

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=self.ignore_index).astype(np.int64, copy=False)

        if valid_path is not None:
            valid_mask = self._read_raster(valid_path)[0] == 1
            if valid_mask.shape != mask.shape:
                raise ValueError(f"Validity mask dimensions differ for {valid_path}")
            mask = mask.copy()
            mask[~valid_mask] = self.ignore_index

        return torch.from_numpy(image), torch.from_numpy(mask)


def create_kuro_siwo_dataloaders(
    root_dir,
    batch_size=8,
    validation_fraction=0.2,
    seed=42,
    num_workers=0,
    image_prefix="MS1",
    mask_prefix="MK0_MLU",
    valid_prefix="MK0_MNA",
    ignore_index=255,
):
    """Create deterministic training and validation loaders for Kuro Siwo patches."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    dataset = KuroSiwoSegmentationDataset(
        root_dir=root_dir,
        image_prefix=image_prefix,
        mask_prefix=mask_prefix,
        valid_prefix=valid_prefix,
        ignore_index=ignore_index,
    )
    if len(dataset) < 2:
        raise ValueError("At least two paired patches are required for a train/validation split")

    validation_size = max(1, round(len(dataset) * validation_fraction))
    validation_size = min(validation_size, len(dataset) - 1)
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    training_set, validation_set = torch.utils.data.random_split(
        dataset, [training_size, validation_size], generator=generator
    )

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return training_loader, validation_loader
