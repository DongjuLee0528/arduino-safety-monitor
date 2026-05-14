import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class HelmetDataset(Dataset):
    """
    Pascal VOC XML 기반 헬멧 감지 데이터셋
    """

    def __init__(self, dataset_path: str, transform=None):
        """
        Args:
            dataset_path: ~/Documents/AIdatasets/helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset/
            transform: 이미지 전처리 변환
        """
        self.dataset_path = os.path.expanduser(dataset_path)
        self.annotations_path = os.path.join(self.dataset_path, "Annotations")
        self.images_path = os.path.join(self.dataset_path, "Images")
        self.transform = transform or self._default_transform()

        # 라벨 매핑
        self.helmet_labels = {"helmet", "head_with_helmet", "person_with_helmet"}
        self.no_helmet_labels = {"head", "person_no_helmet"}

        self.samples = self._load_annotations()

    def _default_transform(self):
        """기본 이미지 전처리: 224x224 리사이즈, 정규화"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_annotations(self) -> List[Tuple[str, int]]:
        """
        XML 파일들을 파싱하여 이미지 경로와 라벨 쌍을 생성

        Returns:
            List of (image_path, label) tuples
            label: 1 (helmet), 0 (no_helmet)
        """
        samples = []

        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations path not found: {self.annotations_path}")

        for xml_file in os.listdir(self.annotations_path):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(self.annotations_path, xml_file)

            # .png 확장자로 이미지 파일명 생성
            image_name = xml_file.replace('.xml', '.png')

            image_path = os.path.join(self.images_path, image_name)

            # 이미지 파일이 존재하는지 확인
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            label = self._parse_xml_label(xml_path)
            samples.append((image_path, label))

        print(f"Loaded {len(samples)} samples")
        return samples

    def _parse_xml_label(self, xml_path: str) -> int:
        """
        XML 파일에서 라벨을 추출하고 분류 규칙 적용

        Args:
            xml_path: XML 파일 경로

        Returns:
            1 (helmet) or 0 (no_helmet)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            helmet_count = 0
            no_helmet_count = 0

            # 모든 object 태그에서 name 추출
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None:
                    class_name = name_elem.text.strip().lower()

                    if class_name in self.helmet_labels:
                        helmet_count += 1
                    elif class_name in self.no_helmet_labels:
                        no_helmet_count += 1

            # 분류 규칙: 두 클래스가 섞이면 no_helmet(0)으로 분류
            if helmet_count > 0 and no_helmet_count > 0:
                return 0  # no_helmet
            elif helmet_count > 0:
                return 1  # helmet
            else:
                return 0  # no_helmet (기본값)

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return 0  # 파싱 실패 시 기본값

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        try:
            # 이미지 로드
            image = Image.open(image_path).convert('RGB')

            # 전처리 적용
            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 오류 발생 시 검은 이미지 반환
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, torch.tensor(0, dtype=torch.long)

    def get_class_distribution(self) -> Dict[str, int]:
        """클래스 분포 반환"""
        helmet_count = sum(1 for _, label in self.samples if label == 1)
        no_helmet_count = len(self.samples) - helmet_count

        return {
            "helmet": helmet_count,
            "no_helmet": no_helmet_count,
            "total": len(self.samples)
        }


def create_data_loaders(dataset_path: str,
                       batch_size: int = 32,
                       train_ratio: float = 0.8,
                       num_workers: int = 0,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    데이터셋을 8:2로 분할하여 train/val DataLoader 생성

    Args:
        dataset_path: 데이터셋 경로
        batch_size: 배치 크기
        train_ratio: 훈련 데이터 비율 (기본 0.8)
        num_workers: DataLoader 워커 수
        pin_memory: GPU 메모리 최적화

    Returns:
        (train_loader, val_loader) tuple
    """
    # 데이터 증강을 위한 훈련용 변환
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 검증용 변환 (데이터 증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 전체 데이터셋을 로드하여 샘플 인덱스 분할
    full_dataset = HelmetDataset(dataset_path, transform=None)

    # 8:2 분할을 위한 인덱스 생성
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    indices = list(range(len(full_dataset)))
    torch.manual_seed(42)  # 재현가능한 분할
    train_indices = torch.randperm(len(full_dataset))[:train_size].tolist()
    val_indices = [i for i in indices if i not in train_indices]

    # 각각 다른 transform으로 Dataset 생성
    train_dataset = HelmetDataset(dataset_path, transform=train_transform)
    val_dataset = HelmetDataset(dataset_path, transform=val_transform)

    # 인덱스를 기반으로 샘플 필터링
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices]

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 클래스 분포 출력
    distribution = full_dataset.get_class_distribution()
    print(f"Dataset distribution: {distribution}")
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    return train_loader, val_loader


if __name__ == "__main__":
    print("=== HelmetDataset Validation ===")

    # MPS 사용 가능 여부 확인
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ MPS available - using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"❌ MPS not available - using device: {device}")

    dataset_path = "~/Documents/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset/"

    try:
        print(f"\n1. Loading dataset from: {os.path.expanduser(dataset_path)}")

        # 전체 데이터셋 로드하여 클래스 분포 확인
        full_dataset = HelmetDataset(dataset_path)
        distribution = full_dataset.get_class_distribution()

        print(f"\n2. Class Distribution:")
        print(f"   - Helmet (1): {distribution['helmet']} samples")
        print(f"   - No Helmet (0): {distribution['no_helmet']} samples")
        print(f"   - Total: {distribution['total']} samples")

        # 이미지 로드 오류 체크
        print(f"\n3. Testing image loading (first 5 samples):")
        error_count = 0
        for i in range(min(5, len(full_dataset))):
            try:
                image, label = full_dataset[i]
                print(f"   Sample {i}: Image shape {image.shape}, Label {label.item()}")
            except Exception as e:
                print(f"   Sample {i}: ❌ Error - {e}")
                error_count += 1

        if error_count == 0:
            print(f"   ✅ All test samples loaded successfully")
        else:
            print(f"   ⚠️  {error_count} samples had loading errors")

        # Train/Val 분할 테스트
        print(f"\n4. Creating train/val data loaders (8:2 split):")
        train_loader, val_loader = create_data_loaders(
            dataset_path=dataset_path,
            batch_size=16,
            train_ratio=0.8
        )

        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        total_samples = train_samples + val_samples
        train_ratio_actual = train_samples / total_samples
        val_ratio_actual = val_samples / total_samples

        print(f"   - Train samples: {train_samples} ({train_ratio_actual:.1%})")
        print(f"   - Val samples: {val_samples} ({val_ratio_actual:.1%})")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")

        # 배치 shape 확인
        print(f"\n5. Batch shape validation:")

        # Train batch 확인
        for images, labels in train_loader:
            print(f"   Train batch - Images: {images.shape}, Labels: {labels.shape}")
            print(f"   Label values in batch: {labels.tolist()}")
            print(f"   Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
            print(f"   Image min/max: {images.min():.3f}/{images.max():.3f}")
            break

        # Val batch 확인
        for images, labels in val_loader:
            print(f"   Val batch - Images: {images.shape}, Labels: {labels.shape}")
            break

        print(f"\n✅ Validation completed successfully!")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()