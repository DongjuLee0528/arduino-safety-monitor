import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class HelmetDataset(Dataset):
    """
    Helmet detection dataset based on Pascal VOC XML format.
    Supports unified SHEL5K + SHWD datasets.

    Each training sample is one per-object crop extracted from the source image
    using the bounding box from the XML annotation.  This matches the runtime
    contract where PersonDetector supplies a whole-person crop to HelmetClassifier.

    One image may produce multiple samples (one per annotated object).
    Images with no recognised objects are skipped entirely.
    XML parse failures are logged and skipped; they do NOT produce a default
    no_helmet label to avoid label contamination.
    """

    # Baseline training contract: whole-person crops only.
    # person_with_helmet → 1 (helmet), person_no_helmet → 0 (no_helmet).
    # helmet / head_with_helmet / head / face are excluded — they are
    # head/object crops that do not match the runtime whole-person input.
    ANNOTATION_LABELS = {
        "person_no_helmet": 0,
        "person_with_helmet": 1,
    }

    # Minimum crop dimension (px) after bbox clipping.
    MIN_DIM = 32

    def __init__(self, shel5k_path: str, shwd_path: str = "", transform=None):
        """
        Args:
            shel5k_path: Path to Safety Helmet Wearing Dataset (Dataset A)
            shwd_path:   Ignored for baseline training (Dataset B excluded).
                         Pass "" or omit; kept for API compatibility.
            transform:   Image preprocessing transformations
        """
        self.transform = transform or self._default_transform()

        # ponytail: kept as instance attrs so existing tests that bypass __init__
        # can still patch them; real filtering uses ANNOTATION_LABELS.
        self.helmet_labels = {"person_with_helmet"}
        self.no_helmet_labels = {"person_no_helmet"}

        shel5k_samples = self._load_dataset(
            annotations_dir=os.path.join(os.path.expanduser(shel5k_path), "Annotations"),
            images_dir=os.path.join(os.path.expanduser(shel5k_path), "Images"),
            image_ext=".png",
            dataset_name="SHEL5K"
        )

        self.samples = shel5k_samples
        if shwd_path:
            print(f"[WARNING] shwd_path supplied but Dataset B is excluded from baseline training. Ignored.")
        print(f"Total samples: {len(self.samples)} (SHEL5K: {len(shel5k_samples)})")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_dataset(self, annotations_dir: str, images_dir: str,
                      image_ext: str, dataset_name: str) -> List[Tuple[str, Tuple[int, int, int, int], int]]:
        """
        Load per-object crop samples from Pascal VOC XML annotations.

        Returns a list of (image_path, bbox_xywh, label) triples.
        """
        samples = []

        if not os.path.exists(annotations_dir):
            raise FileNotFoundError(f"[{dataset_name}] Annotations not found: {annotations_dir}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"[{dataset_name}] Images not found: {images_dir}")

        for xml_file in os.listdir(annotations_dir):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(annotations_dir, xml_file)
            image_name = xml_file.replace('.xml', image_ext)
            image_path = os.path.join(images_dir, image_name)

            if not os.path.exists(image_path):
                print(f"[{dataset_name}] Warning: Image not found: {image_path}")
                continue

            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except Exception as e:
                print(f"[{dataset_name}] Warning: Image unreadable: {image_path}: {e}")
                continue

            for bbox, label in self._parse_xml_objects(xml_path):
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                w = min(image_width - x, w)
                h = min(image_height - y, h)
                if w <= 0 or h <= 0:
                    continue
                if w < self.MIN_DIM or h < self.MIN_DIM:
                    continue
                samples.append((image_path, (x, y, w, h), label))

        print(f"[{dataset_name}] Loaded {len(samples)} samples")
        return samples

    def _parse_xml_objects(self, xml_path: str) -> List[Tuple[Tuple[int, int, int, int], int]]:
        """
        Parse a Pascal VOC XML file and return (bbox_xywh, label) for every
        recognised object.

        Unknown object classes are silently skipped.
        Malformed or missing bndbox elements are silently skipped.
        On XML parse failure the error is logged and an empty list is returned
        so that the broken file does not produce spurious no_helmet labels.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None or not name_elem.text:
                    continue
                class_name = name_elem.text.strip().lower()
                if class_name not in self.ANNOTATION_LABELS:
                    continue
                label = self.ANNOTATION_LABELS[class_name]

                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                try:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                except (TypeError, ValueError, AttributeError):
                    continue

                if xmin >= xmax or ymin >= ymax:
                    continue

                objects.append(((xmin, ymin, xmax - xmin, ymax - ymin), label))
            return objects
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bbox, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            x, y, w, h = bbox
            x1 = max(0, min(x, image.width))
            y1 = max(0, min(y, image.height))
            x2 = max(x1, min(x + w, image.width))
            y2 = max(y1, min(y + h, image.height))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"empty bbox {bbox} for image {image_path}")
            image = image.crop((x1, y1, x2, y2))
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            raise RuntimeError(f"Error loading training sample {image_path}: {e}") from e

    def get_class_distribution(self) -> Dict[str, int]:
        helmet_count = sum(1 for _, _, label in self.samples if label == 1)
        no_helmet_count = len(self.samples) - helmet_count
        return {
            "helmet": helmet_count,
            "no_helmet": no_helmet_count,
            "total": len(self.samples)
        }


def _split_source_groups(samples: List[Tuple[str, Tuple[int, int, int, int], int]],
                         train_ratio: float) -> Tuple[List[int], List[int]]:
    """Split samples by source image so one image cannot leak across splits."""
    source_paths = sorted({sample[0] for sample in samples})
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(len(source_paths), generator=generator).tolist()
    train_source_count = int(train_ratio * len(source_paths))
    train_sources = {source_paths[i] for i in order[:train_source_count]}
    train_indices = [i for i, sample in enumerate(samples) if sample[0] in train_sources]
    val_indices = [i for i, sample in enumerate(samples) if sample[0] not in train_sources]
    return train_indices, val_indices


def create_data_loaders(shel5k_path: str,
                        shwd_path: str = "",
                        batch_size: int = 32,
                        train_ratio: float = 0.8,
                        num_workers: int = 0,
                        pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Split dataset 8:2 and create train/val DataLoaders.

    Args:
        shel5k_path: Path to SHEL5K dataset
        shwd_path: Path to SHWD dataset
        batch_size: Batch size
        train_ratio: Training data ratio (default 0.8)
        num_workers: Number of DataLoader workers
        pin_memory: GPU memory optimization

    Returns:
        (train_loader, val_loader) tuple
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = HelmetDataset(shel5k_path, shwd_path, transform=None)

    train_indices, val_indices = _split_source_groups(full_dataset.samples, train_ratio)
    train_size = len(train_indices)
    val_size = len(val_indices)

    train_dataset = HelmetDataset(shel5k_path, shwd_path, transform=train_transform)
    val_dataset = HelmetDataset(shel5k_path, shwd_path, transform=val_transform)

    train_sources = {full_dataset.samples[i][0] for i in train_indices}
    val_sources = {full_dataset.samples[i][0] for i in val_indices}
    train_dataset.samples = [sample for sample in train_dataset.samples if sample[0] in train_sources]
    val_dataset.samples = [sample for sample in val_dataset.samples if sample[0] in val_sources]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    distribution = full_dataset.get_class_distribution()
    print(f"Dataset distribution: {distribution}")
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    return train_loader, val_loader


from mpu.config import SHEL5K_PATH, SHWD_PATH

if __name__ == "__main__":
    print("=== HelmetDataset Validation ===")
    print(f"PyTorch version: {torch.__version__}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"MPS available - using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"MPS not available - using device: {device}")

    try:
        print(f"\n1. Loading datasets...")
        full_dataset = HelmetDataset(SHEL5K_PATH, SHWD_PATH)
        distribution = full_dataset.get_class_distribution()

        print(f"\n2. Class Distribution:")
        print(f"   - Helmet (1):    {distribution['helmet']} samples")
        print(f"   - No Helmet (0): {distribution['no_helmet']} samples")
        print(f"   - Total:         {distribution['total']} samples")

        print(f"\n3. Testing image loading (first 5 samples):")
        error_count = 0
        for i in range(min(5, len(full_dataset))):
            try:
                image, label = full_dataset[i]
                print(f"   Sample {i}: shape={image.shape}, label={label.item()}")
            except Exception as e:
                print(f"   Sample {i}: Error - {e}")
                error_count += 1

        if error_count == 0:
            print(f"   All test samples loaded successfully")

        print(f"\n4. Creating train/val loaders (8:2 split)...")
        train_loader, val_loader = create_data_loaders(
            shel5k_path=SHEL5K_PATH,
            shwd_path=SHWD_PATH,
            batch_size=16
        )
        print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        print(f"\n5. Batch shape validation:")
        for images, labels in train_loader:
            print(f"   Train - Images: {images.shape}, Labels: {labels.shape}")
            print(f"   Image min/max: {images.min():.3f}/{images.max():.3f}")
            break
        for images, labels in val_loader:
            print(f"   Val   - Images: {images.shape}, Labels: {labels.shape}")
            break

        print(f"\nValidation completed successfully!")

    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
