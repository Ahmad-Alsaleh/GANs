import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
import random
from PIL import Image
import argparse

class SketchDataset(Dataset):
    def __init__(self, bad_sketch_dir, good_sketch_dir):
        self.bad_sketch_paths = sorted(
            [os.path.join(bad_sketch_dir, x) for x in os.listdir(bad_sketch_dir)]
        )
        self.good_sketch_paths = sorted(
            [os.path.join(good_sketch_dir, x) for x in os.listdir(good_sketch_dir)]
        )
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.transform_augment = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(10, shear=10, scale=(0.8, 1.2), fill=255),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __getitem__(self, index):
        bad_sketch = Image.open(
            self.bad_sketch_paths[index % len(self.bad_sketch_paths)]
        )

        good_sketch = Image.open(random.choice(self.good_sketch_paths))

        bad_sketch = self.transform(bad_sketch)
        good_sketch = self.transform(good_sketch)

        return {"A": bad_sketch, "B": good_sketch}

    def __len__(self):
        return len(self.bad_sketch_paths)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)
    

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Downsampling blocks
        self.down_blocks = nn.Sequential(
            self._make_layer(64, 128), self._make_layer(128, 256)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(9)])

        # Upsampling blocks
        self.up_blocks = nn.Sequential(
            self._make_layer(256, 128, upsample=True),
            self._make_layer(128, 64, upsample=True),
        )

        # Output convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3), nn.Tanh()
        )

    def _make_layer(self, in_channels, out_channels, upsample=False):
        if upsample:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        return self.conv2(x)
    
parser = argparse.ArgumentParser()
parser.add_argument("--bad-sketch-dir", type=str, required=True)
parser.add_argument("--good-sketch-dir", type=str, required=True)
parser.add_argument("--class-type", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--num-images", type=int, default=5)
parser.add_argument("--checkpoint", type=str, required=True)

args = parser.parse_args()

class_type = args.class_type
bad_sketch_dir = os.path.join(args.bad_sketch_dir, class_type)
good_sketch_dir = os.path.join(args.good_sketch_dir, class_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()
dataset = SketchDataset(bad_sketch_dir, good_sketch_dir)
with torch.no_grad():
    for _ in range(args.num_images):
        index = random.randint(0, len(dataset))
        sample = dataset[index]
        bad_sketch = sample["A"].unsqueeze(0).to(device)
        fake_good_sketch = model(bad_sketch)
        os.makedirs(args.output_dir, exist_ok=True)
        save_image(
            fake_good_sketch, f"{args.output_dir}/{class_type}_{index}.png"
        )
        
print("Images saved")