"""Train a U-Net on paired Kuro Siwo Sentinel-1 patches and flood labels."""

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from model import UNet


EXPECTED_LAYOUT = (
    "Expected layout (one sample per folder):\n"
    "  <root_folder>/<sample>/MS1*.tif or *.tiff\n"
    "  <root_folder>/<sample>/MK0_MLU*.tif or *.tiff\n"
    "  <root_folder>/<sample>/MK0_MNA*.tif or *.tiff  (optional validity mask)"
)


def validate_dataset_root(root_folder):
    root = Path(root_folder).expanduser()
    if not root.exists():
        raise SystemExit(
            f"Error: dataset root folder does not exist: {root.resolve()}\n{EXPECTED_LAYOUT}"
        )
    if not root.is_dir():
        raise SystemExit(
            f"Error: dataset root must be a folder, not a file: {root.resolve()}\n{EXPECTED_LAYOUT}"
        )

    rasters_by_folder = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            rasters_by_folder.setdefault(path.parent, []).append(path)

    paired_samples = [
        (folder, paths)
        for folder, paths in rasters_by_folder.items()
        if any(path.stem.startswith("MS1") for path in paths)
        and any(path.stem.startswith("MK0_MLU") for path in paths)
    ]
    if not paired_samples:
        raise SystemExit(
            f"Error: no paired MS1 imagery and MK0_MLU labels were found under {root.resolve()}.\n"
            f"{EXPECTED_LAYOUT}"
        )
    return root, paired_samples


def resolve_channel_names(value, in_channels):
    names = [name.strip() for name in value.split(",") if name.strip()]
    if len(names) != in_channels:
        raise ValueError(
            f"--channel-names specifies {len(names)} names, but the training rasters "
            f"have {in_channels} channels."
        )
    if len(set(name.lower() for name in names)) != len(names):
        raise ValueError("--channel-names must contain unique names in raster band order.")
    return names


def inspect_dataset(train_loader, val_loader, ignore_index):
    """Read masks to discover raw label IDs and count training pixels per class."""
    sample_images, _ = next(iter(train_loader))
    if sample_images.ndim != 4:
        raise ValueError(
            f"Expected image batches shaped [batch, channels, height, width], got {sample_images.shape}"
        )
    in_channels = sample_images.shape[1]

    all_counts = Counter()
    train_counts = Counter()
    for loader, is_training in ((train_loader, True), (val_loader, False)):
        for _, masks in loader:
            values, pixel_counts = torch.unique(masks, return_counts=True)
            for value, pixel_count in zip(values.tolist(), pixel_counts.tolist()):
                if value == ignore_index:
                    continue
                if value < 0:
                    raise ValueError(f"Mask labels must be non-negative; found {value}")
                raw_value = int(value)
                all_counts[raw_value] += int(pixel_count)
                if is_training:
                    train_counts[raw_value] += int(pixel_count)

    class_values = sorted(all_counts)
    if not class_values:
        raise ValueError("No valid class pixels found; check masks and ignore_index")

    print(
        f"Observed mask values: {class_values}; ignore_index={ignore_index}; "
        f"num_classes={len(class_values)}"
    )
    print(
        "Class mapping: "
        + ", ".join(f"raw {value} -> model {index}" for index, value in enumerate(class_values))
    )
    return in_channels, class_values, train_counts


def remap_masks(masks, class_values, ignore_index):
    """Map arbitrary non-negative label IDs to contiguous CrossEntropy targets."""
    targets = torch.full_like(masks, ignore_index)
    for model_index, raw_value in enumerate(class_values):
        targets[masks == raw_value] = model_index
    return targets


def make_class_weights(train_counts, class_values, imbalance_threshold, device):
    """Return inverse-frequency weights only when training pixels are imbalanced."""
    counts = torch.tensor([train_counts[value] for value in class_values], dtype=torch.float32)
    if torch.any(counts == 0):
        missing = [value for value in class_values if train_counts[value] == 0]
        raise ValueError(
            f"Classes {missing} have no training pixels. Adjust the validation split or seed."
        )

    imbalance_ratio = (counts.max() / counts.min()).item()
    if imbalance_ratio < imbalance_threshold:
        print(f"Class pixel imbalance ratio {imbalance_ratio:.1f}; using unweighted loss")
        return None

    frequencies = counts / counts.sum()
    weights = frequencies.rsqrt()
    weights = weights / weights.mean()
    print(
        f"Class pixel imbalance ratio {imbalance_ratio:.1f}; "
        f"using weights {weights.tolist()}"
    )
    return weights.to(device)


def update_confusion_matrix(confusion, logits, targets, ignore_index):
    num_classes = confusion.shape[0]
    predictions = logits.argmax(dim=1)
    valid = targets != ignore_index
    if not torch.any(valid):
        return
    encoded = targets[valid] * num_classes + predictions[valid]
    confusion += torch.bincount(
        encoded, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def class_metrics(confusion):
    true_positive = confusion.diag().float()
    actual = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    union = actual + predicted - true_positive
    f1_denominator = actual + predicted

    iou = torch.where(union > 0, true_positive / union, torch.nan)
    f1 = torch.where(
        f1_denominator > 0,
        2 * true_positive / f1_denominator,
        torch.nan,
    )
    return iou, f1


def train_epoch(model, loader, optimizer, criterion, device, class_values, ignore_index, max_batches):
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch_index, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        targets = remap_masks(masks.to(device), class_values, ignore_index)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    if batch_count == 0:
        raise ValueError("The training loader produced no batches")
    return total_loss / batch_count


@torch.no_grad()
def validate(model, loader, criterion, device, class_values, ignore_index, max_batches):
    model.eval()
    confusion = torch.zeros(
        (len(class_values), len(class_values)), dtype=torch.int64, device=device
    )
    total_loss = 0.0
    batch_count = 0

    for batch_index, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        targets = remap_masks(masks.to(device), class_values, ignore_index)
        logits = model(images)
        total_loss += criterion(logits, targets).item()
        batch_count += 1
        update_confusion_matrix(confusion, logits, targets, ignore_index)

    if batch_count == 0:
        raise ValueError("The validation loader produced no batches")
    iou, f1 = class_metrics(confusion)
    return total_loss / batch_count, iou.cpu(), f1.cpu(), confusion.cpu()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'Example: python train.py "C:\\data\\KuroSiwo" --flood-class 2 '
            "--epochs 1 --max-batches 5\n" + EXPECTED_LAYOUT
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root_folder", help="Folder containing the Kuro Siwo dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--flood-class", type=int, required=True, help="Raw mask value for flood")
    parser.add_argument(
        "--channel-names",
        default="VV,VH",
        help="Comma-separated training input channel names in raster order (default: VV,VH)",
    )
    parser.add_argument(
        "--input-scale",
        choices=("linear", "db"),
        default="linear",
        help="Input scale used by the training rasters; this loader passes pixels through unchanged",
    )
    parser.add_argument("--imbalance-threshold", type=float, default=10.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit both training and validation to this many batches per epoch",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers cannot be negative")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    if args.imbalance_threshold <= 0:
        parser.error("--imbalance-threshold must be positive")
    return args


def main():
    args = parse_args()
    from training_data import create_kuro_siwo_dataloaders

    dataset_root, paired_samples = validate_dataset_root(args.root_folder)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Paired MS1/MK0_MLU samples found: {len(paired_samples)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_kuro_siwo_dataloaders(
        root_dir=str(dataset_root),
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        ignore_index=args.ignore_index,
    )

    in_channels, class_values, train_counts = inspect_dataset(
        train_loader, val_loader, args.ignore_index
    )
    channel_names = resolve_channel_names(args.channel_names, in_channels)
    if args.flood_class not in class_values:
        raise ValueError(
            f"Flood class {args.flood_class} was not found. Observed classes: {class_values}"
        )
    flood_index = class_values.index(args.flood_class)

    # CrossEntropyLoss needs contiguous target IDs, so remap the observed raw labels.
    class_weights = make_class_weights(
        train_counts, class_values, args.imbalance_threshold, device
    )
    model = UNet(
        in_channels=in_channels,
        num_classes=len(class_values),
        base_channels=args.base_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=args.ignore_index
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_flood_iou = float("-inf")

    for epoch in range(1, args.epochs + 1):
        training_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            class_values,
            args.ignore_index,
            args.max_batches,
        )
        validation_loss, iou, f1, confusion = validate(
            model,
            val_loader,
            criterion,
            device,
            class_values,
            args.ignore_index,
            args.max_batches,
        )

        class_summary = " | ".join(
            f"raw {raw_value}: IoU={iou[index].item():.4f}, F1={f1[index].item():.4f}"
            for index, raw_value in enumerate(class_values)
        )
        flood_iou = iou[flood_index].item()
        print(
            f"Epoch {epoch}/{args.epochs} | train loss={training_loss:.4f} "
            f"| val loss={validation_loss:.4f} | flood IoU={flood_iou:.4f}"
        )
        print(f"  Validation per class: {class_summary}")

        # A flood class absent from the evaluated validation batches has no measurable IoU.
        flood_union = confusion[flood_index].sum() + confusion[:, flood_index].sum()
        if flood_union == 0:
            print("  Flood class absent from evaluated validation pixels; checkpoint not updated")
            continue
        if flood_iou > best_flood_iou:
            best_flood_iou = flood_iou
            checkpoint_path = checkpoint_dir / "best_unet.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "in_channels": in_channels,
                    "input_channel_names": channel_names,
                    "input_scale": args.input_scale,
                    "num_classes": len(class_values),
                    "class_values": class_values,
                    "ignore_index": args.ignore_index,
                    "flood_class": args.flood_class,
                    "validation_flood_iou": best_flood_iou,
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()