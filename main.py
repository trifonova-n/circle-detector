import matplotlib
matplotlib.use('Agg')
import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
import torch.nn.functional as F


from torch_lr_finder import LRFinder
from torchvision.transforms import functional as trF

import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import time

global_model = None
batch_size = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    def __init__(self, length, transforms=None, seed=10):
        self.seed = seed
        self.length = length
        self.transforms = transforms
        self.images = []
        self.circles = []
        self._generate()

    def _generate(self):
        np.random.seed(self.seed)
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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    lr = 0.02
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    loss_function = F.mse_loss
    step = 0
    train_losses = []
    train_acc = []
    for current_epoch in range(epochs):
        print("epoch", current_epoch)
        model.train()
        tic = time.time()
        for (x, target) in iter(train_loader):
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = loss_function(target, output)
            train_losses.append(float(loss))
            train_acc.extend(accuracy(output, target))
            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            step += 1

            # time step duration:
            n = 10
            if step % n == 0:
                toc = time.time()

                metric = np.mean(np.array(train_acc) > 0.7)
                print("one training step does take approximately " + str((toc - tic) * (1.0/n)) + " seconds)")
                print(f"Step: {step} Training loss {np.mean(train_losses)}, accuracy {np.mean(train_acc)}, metric {metric}")
                train_losses = []
                train_acc = []

        valid_losses = []
        valid_acc = []
        # Validation step
        model.eval()
        for (x, target) in iter(valid_loader):
            x = x.to(device)
            target = target.to(device)
            with torch.no_grad():
                output = model(x)
                loss = loss_function(target, output)
                valid_losses.append(float(loss))
                valid_acc.extend(accuracy(output, target))
        metric = np.mean(np.array(valid_acc) > 0.7)
        print(f"Step {step} Validation loss {np.mean(valid_losses)}, accuracy {np.mean(valid_acc)}, metric {metric}")


def accuracy(input, target):
    ious = []
    for i, t in zip(input, target):
        ious.append(iou(i, t))
    return np.array(ious)


def find_circle(img):
    model = global_model
    model.eval()
    img = trF.to_tensor(img.astype(np.float32)).to(device)
    with torch.no_grad():
        output = model(img[None,...])
    # Fill in this function
    return output[0]


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def find_lr(model, train_data):
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)  # to inspect the loss-learning rate graph
    fig.savefig('learning_rate.png', dpi=fig.dpi)
    lr_finder.reset()  # to reset the model and optimizer to their initial state


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
    parser.add_argument('--find-lr', action='store_true', help='Find learning rate')
    parser.add_argument('--model', default='circle-detector.pth')
    parser.add_argument('--data-size', default=5000, help='Training data size')
    args = parser.parse_args()
    valid_data_size = args.data_size // 10
    valid_data = CircleDataset(valid_data_size, seed=20)
    train_data = CircleDataset(args.data_size, seed=30)
    model = FindCircleNet()
    print(model)
    if args.find_lr:
        find_lr(model, train_data)
        exit(0)
    if args.train:
        train(model, train_data, valid_data, epochs=100)
        torch.save(model.state_dict(), args.model)
        model.eval()
    else:
        model.load_state_dict(torch.load(args.model))
        model.to(device)
        model.eval()

    global_model = model

    main()

