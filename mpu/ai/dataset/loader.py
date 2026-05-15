import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class HelmetDataset(Dataset):
    """
    Pascal VOC XML 기반 헬멧 감지 데이터셋
    SHEL5K + SHWD 두 데이터셋 통합 지원
    """

    def __init__(self, shel5k_path: str, shwd_path: str, transform=None):
        """
        Args:
            shel5k_path: Safety Helmet Wearing Dataset 경로
            shwd_path:   VOC2028 경로
            transform:   이미지 전처리 변환
        """
        self.transform = transform or self._default_transform()

        self.helmet_labels = {"helmet", "head_with_helmet", "person_with_helmet"}
        self.no_helmet_labels = {"head", "person_no_helmet"}

        shel5k_samples = self._load_dataset(
            annotations_dir=os.path.join(os.path.expanduser(shel5k_path), "Annotations"),
            images_dir=os.path.join(os.path.expanduser(shel5k_path), "Images"),
            image_ext=".png",
            dataset_name="SHEL5K"
        )
        shwd_samples = self._load_dataset(
            annotations_dir=os.path.join(os.path.expanduser(shwd_path), "Annotations"),
            images_dir=os.path.join(os.path.expanduser(shwd_path), "JPEGImages"),
            image_ext=".jpg",
            dataset_name="SHWD"
        )

        self.samples = shel5k_samples + shwd_samples
        print(f"Total samples: {len(self.samples)} (SHEL5K: {len(shel5k_samples)}, SHWD: {len(shwd_samples)})")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_dataset(self, annotations_dir: str, images_dir: str,
                      image_ext: str, dataset_name: str) -> List[Tuple[str, int]]:
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

            label = self._parse_xml_label(xml_path)
            samples.append((image_path, label))

        print(f"[{dataset_name}] Loaded {len(samples)} samples")
        return samples

    def _parse_xml_label(self, xml_path: str) -> int:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            helmet_count = 0
            no_helmet_count = 0

            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None:
                    class_name = name_elem.text.strip().lower()
                    if class_name in self.helmet_labels:
                        helmet_count += 1
                    elif class_name in self.no_helmet_labels:
                        no_helmet_count += 1

            if helmet_count > 0 and no_helmet_count > 0:
                return 0
            elif helmet_count > 0:
                return 1
            else:
                return 0

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, 224, 224), torch.tensor(0, dtype=torch.long)

    def get_class_distribution(self) -> Dict[str, int]:
        helmet_count = sum(1 for _, label in self.samples if label == 1)
        no_helmet_count = len(self.samples) - helmet_count
        return {
            "helmet": helmet_count,
            "no_helmet": no_helmet_count,
            "total": len(self.samples)
        }


def create_data_loaders(shel5k_path: str,
                        shwd_path: str,
                        batch_size: int = 32,
                        train_ratio: float = 0.8,
                        num_workers: int = 0,
                        pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
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

    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    indices = list(range(len(full_dataset)))
    torch.manual_seed(42)
    train_indices = torch.randperm(len(full_dataset))[:train_size].tolist()
    val_indices = [i for i in indices if i not in set(train_indices)]

    train_dataset = HelmetDataset(shel5k_path, shwd_path, transform=train_transform)
    val_dataset = HelmetDataset(shel5k_path, shwd_path, transform=val_transform)

    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    distribution = full_dataset.get_class_distribution()
    print(f"Dataset distribution: {distribution}")
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    return train_loader, val_loader


SHEL5K_PATH = "~/Documents/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset"
SHWD_PATH   = "~/Documents/AIdatasets/ helmet-safety-robot/raw/VOC2028"

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