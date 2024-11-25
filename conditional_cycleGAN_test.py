import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from torchvision.utils import save_image
from pathlib import Path
from collections import defaultdict
import argparse
import os

class SketchDataset(Dataset):
    def __init__(self, bad_root, good_root, samples_per_class=None, fraction=None):
        self.bad_root = Path(bad_root)
        self.good_root = Path(good_root)
        self.transform_bad = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.transform_augmented = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(
                    10,
                    shear=10,
                    scale=(0.8, 1.2),
                    fill=255,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        # Get all classes (folders)
        self.classes = sorted([d.name for d in self.bad_root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths and their classes
        self.bad_images = []
        self.good_images = []

        for class_name in self.classes:
            bad_class_path = self.bad_root / class_name
            good_class_path = self.good_root / class_name

            bad_class_images = list(bad_class_path.glob("*.png"))
            good_class_images = list(good_class_path.glob("*.png"))

            class_idx = self.class_to_idx[class_name]

            self.bad_images.extend([(str(img), class_idx) for img in bad_class_images])
            self.good_images.extend(
                [(str(img), class_idx) for img in good_class_images]
            )

        # Ensure we have images
        assert len(self.bad_images) > 0, "No bad sketches found"
        assert len(self.good_images) > 0, "No good sketches found"

        if samples_per_class is not None or fraction is not None:
            # Group bad images by class
            class_to_images = defaultdict(list)
            for img_path, class_idx in self.bad_images:
                class_to_images[class_idx].append((img_path, class_idx))

            # Create subset based on specified criterion
            subset_images = []
            for class_idx, images in class_to_images.items():
                if samples_per_class is not None:
                    # Take exact number of samples per class
                    n_samples = min(samples_per_class, len(images))
                elif fraction is not None:
                    # Take fraction of samples per class
                    n_samples = max(1, int(len(images) * fraction))
                else:
                    n_samples = len(images)

                # Randomly sample from this class
                subset_images.extend(random.sample(images, n_samples))

            # Replace bad_images with subset
            self.bad_images = subset_images

    def __getitem__(self, index):
        # Get a bad sketch
        bad_path, bad_class = self.bad_images[index % len(self.bad_images)]

        # Randomly select a good sketch from the same class
        good_candidates = [
            (path, cls) for path, cls in self.good_images if cls == bad_class
        ]
        good_path, good_class = random.choice(good_candidates)

        # Load and transform images
        bad_img = Image.open(bad_path)
        good_img = Image.open(good_path)

        bad_img = self.transform_bad(bad_img)
        good_img = self.transform_augmented(good_img)

        return {"bad": bad_img, "good": good_img, "class": torch.tensor(bad_class)}

    def __len__(self):
        return len(self.bad_images)

    def get_class_distribution(self):
        """
        Returns the distribution of samples across classes.

        Returns:
            dict: Dictionary mapping class names to number of samples
        """
        distribution = defaultdict(int)
        for _, class_idx in self.bad_images:
            class_name = self.classes[class_idx]
            distribution[class_name] += 1
        return dict(distribution)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.num_classes = num_classes

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 64)

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(1 + 64, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(6)])

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, padding=3), nn.Tanh()
        )

    def forward(self, x, class_labels):
        # Embed class labels and expand to image dimensions
        class_embed = self.class_embedding(class_labels)
        class_embed = class_embed.view(class_embed.size(0), -1, 1, 1)
        class_embed = class_embed.expand(-1, -1, x.size(2), x.size(3))

        # Concatenate input with class embedding
        x = torch.cat([x, class_embed], dim=1)

        # Forward pass through the network
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.up1(x)
        x = self.up2(x)
        return self.output(x)

parser = argparse.ArgumentParser()
parser.add_argument("--bad-sketch-dir", type=str, required=True)
parser.add_argument("--good-sketch-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--num-samples", type=int, default=5)
parser.add_argument("--checkpoint-path", type=str, required=True)
args = parser.parse_args()

# Load the dataset
dataset = SketchDataset(
    args.bad_sketch_dir, args.good_sketch_dir, samples_per_class=3000
)

model = Generator(num_classes=12)
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint["G_bad_to_good_state_dict"])
model.eval()
with torch.no_grad():
    for i in range(args.num_samples):
        random_idx = random.randint(0, len(dataset))
        sample = dataset[random_idx]
        bad_img = sample["bad"].unsqueeze(0)
        class_label = sample["class"].unsqueeze(0)
        output = model(bad_img, class_label)
        os.makedirs(args.output_dir, exist_ok=True)
        save_image(output, os.path.join(args.output_dir, f"output_{i}.png"))
print("Images saved")