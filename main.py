import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
import torch.nn.functional as F

import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import time


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


class CircleDataset(torch.utils.data.Dataset):
    def __init__(self, length, transforms=None):
        self.length = length
        self.transforms = transforms
        self.images = []
        self.circles = []
        self._generate()

    def _generate(self):
        np.random.seed(10)
        for i in range(self.length):
            circle, img = noisy_circle(200, 50, 2)
            self.images.append(img.astype(np.float32))
            self.circles.append(np.array(circle, dtype=np.float32))

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.circles[idx]
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img[None, ...], target

    def __len__(self):
        return len(self.images)


class FindCircleNet(torch.nn.Module):
    def __init__(self):
        super(FindCircleNet, self).__init__()
        self.res_net = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=3)

        self.res_net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)

    def forward(self, x):
        return self.res_net.forward(x)


def train(model, train_data, valid_data, epochs=10):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda')# if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    batch_size = 64
    lr = 0.001
    weight_decay = 0
    clip = None
    model.to(device)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = F.mse_loss
    model.train()
    step = 0
    train_losses = []
    for current_epoch in range(epochs):
        print("epoch", current_epoch)
        tic = time.time()
        for (x, target) in iter(train_loader):
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = loss_function(target, output)
            train_losses.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            step += 1

            # time step duration:
            n = 10
            if step % n == 0:
                toc = time.time()
                print("one training step does take approximately " + str((toc - tic) * (1.0/n)) + " seconds)")
                print(f"Step: {step} Training loss {np.mean(train_losses)}")
                train_losses = []

        valid_losses = []
        # Validation step
        for (x, target) in iter(valid_loader):
            x = x.to(device)
            target = target.to(device)
            with torch.no_grad():
                output = model(x)
                loss = loss_function(target, output)
                valid_losses.append(float(loss))
        print(f"Step {step} Validation loss {np.mean(valid_losses)}")





def find_circle(img):
    # Fill in this function
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model or run trained model')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--model', default='circle-detector.pth')
    parser.add_argument('--data-size', default=1000, help='Training data size')
    args = parser.parse_args()
    valid_data_size = args.data_size // 10
    valid_data = CircleDataset(valid_data_size)
    model = FindCircleNet()
    print(model)
    if args.train:
        train_data = CircleDataset(args.data_size)
        train(model, train_data, valid_data, epochs=100)
        model.eval()
    else:
        model.load_state_dict(args.model)
        model.eval()
    main()

