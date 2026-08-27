import os
import cv2
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RESIDEOUTDataset(Dataset):
    def __init__(self, base_dir, split, target_size=(256,256), max_samples=None):
        self.samples = []
        self.target_size = target_size

        hazy_dir = os.path.join(base_dir, split, "hazy")
        gt_dir = os.path.join(base_dir, split, "GT")

        print(f"\n Loading {split} dataset (FULL DATASET)...")

        # STEP 1: Load all filenames
        hazy_images = os.listdir(hazy_dir)

        # STEP 2: Optional limit (only if needed)
        if max_samples is not None:
            hazy_images = random.sample(hazy_images, min(max_samples, len(hazy_images)))

        # STEP 3: Build dataset
        for img in tqdm(hazy_images, desc=f"{split.upper()} Progress", unit="images"):
            hazy_path = os.path.join(hazy_dir, img)
            gt_path = os.path.join(gt_dir, img)

            if os.path.exists(gt_path):
                self.samples.append((hazy_path, gt_path))

        print(f"✅ {split.upper()} Loaded: {len(self.samples)} samples")

        # ✅ FIXED TRANSFORM (VERY IMPORTANT)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hazy_path, gt_path = self.samples[idx]

        hazy = cv2.imread(hazy_path)
        gt = cv2.imread(gt_path)

        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        hazy = self.transform(hazy)
        gt = self.transform(gt)

        return hazy, gt, hazy_path
