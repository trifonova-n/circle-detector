import logging
logging.basicConfig(filename='output.txt', filemode='w', format=' %(message)s', level=logging.INFO)

from typing import Optional, Tuple, Callable
import matplotlib
matplotlib.use('Agg')
import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import time

from torch_lr_finder import LRFinder
from torchvision.transforms import functional as trF
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
# ============================= #
#       Original script         #
# ============================= #


# passing model to original script via this variable
global_model: Optional['FindCircleNet'] = None


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


def find_circle(img):
    # Fill in this function
    model = global_model
    model.eval()
    img = trF.to_tensor(img.astype(np.float32)).to(device)
    with torch.no_grad():
        output = model(img[None, ...])
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


def main():
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())


# ============================= #
#        Training code          #
# ============================= #

batch_size = 64
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_checkpoint = 'circle-detector.pth'


class CircleDataset(torch.utils.data.Dataset):
    def __init__(self, length: int, transforms: Optional[Callable] = None, seed: int = 10):
        super().__init__()
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

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        img = self.images[idx]
        target = self.circles[idx]
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img[np.newaxis, ...], target

    def __len__(self):
        return len(self.images)


class FindCircleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # reduce model size to be less than 1M params
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [2, 24, 2, 2],
            [2, 32, 2, 2],
            [2, 64, 3, 2],
            [2, 96, 3, 1],
            [2, 160, 3, 2],
            [2, 320, 1, 1],
        ]
        # only loads model definition
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False, num_classes=3,
                                    inverted_residual_setting=inverted_residual_setting)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.model.forward(x)


def train(model: FindCircleNet, train_data: CircleDataset, valid_data: CircleDataset, epochs: int = 10):
    lr = 0.02
    weight_decay = 0
    clip = None
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
    train_iou = []
    for current_epoch in range(epochs):
        logger.info(f"epoch {current_epoch}")
        if current_epoch < 0.98*epochs:
            model.train()
        tic = time.time()
        for (x, target) in iter(train_loader):
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            loss = loss_function(target, output)
            train_losses.append(float(loss))
            train_iou.extend(iou_metric(output, target))
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

                metric = np.mean(np.array(train_iou) > 0.7)
                logger.debug(f"one training step does take approximately {(toc - tic) * (1.0/n)} seconds)")
                logger.info(f"Step: {step} Training loss {np.mean(train_losses):.4f}, iou {np.mean(train_iou):.4f}, metric {metric:.4f}")
                train_losses = []
                train_iou = []

        valid_losses = []
        valid_iou = []
        # Validation step
        model.eval()
        for x, target in valid_loader:
            x = x.to(device)
            target = target.to(device)
            with torch.no_grad():
                output = model(x)
                loss = loss_function(target, output)
                valid_losses.append(float(loss))
                valid_iou.extend(iou_metric(output, target))
        metric = np.mean(np.array(valid_iou) > 0.7)
        logger.info(f"Step: {step} Validation loss {np.mean(valid_losses):.4f}, iou {np.mean(valid_iou):.4f}, metric {metric:.4f}")


def iou_metric(input: np.ndarray, target: np.ndarray):
    ious = []
    for i, t in zip(input, target):
        ious.append(iou(i, t))
    return np.array(ious)


def find_lr(model: torch.nn.Module, train_data: CircleDataset):
    # range test for finding learning rate as described in
    # https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6
    lr_image = 'learning_rate.png'
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    logger.info("Running range test for learning rate")
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)  # to inspect the loss-learning rate graph
    logger.info(f"Saving image with learning rate plot to {lr_image}")
    fig.savefig(lr_image, dpi=fig.dpi)
    lr_finder.reset()  # to reset the model and optimizer to their initial state


def eval_model(model: FindCircleNet):
    model.eval()
    # trying to retain structure of original program
    global global_model
    global_model = model
    main()


def load_data(data_size: int) -> Tuple[CircleDataset, CircleDataset]:
    valid_data = CircleDataset(data_size // 10, seed=20)
    train_data = CircleDataset(data_size, seed=30)
    return train_data, valid_data


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(checkpoint: Optional[str] = None) -> FindCircleNet:
    model = FindCircleNet()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    logger.debug("Model info:")
    logger.debug(str(model))

    logger.info(f"Number of model  parameters: {count_parameters(model)}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model or run trained model')
    parser.add_argument('command', choices=['find-lr', 'train', 'eval'], help="Use find-lr to run range test and save plot to image, train "
                                                                              "to train the model or eval to run evaluation")
    parser.add_argument('--data-size', default=10000, help='Training data size')
    args = parser.parse_args()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    logger.info(f"Using device: {device}")
    train_data, valid_data = load_data(args.data_size)
    model = create_model(checkpoint=None if args.command != 'eval' else model_checkpoint)

    if args.command == 'find-lr':
        find_lr(model, train_data)
    elif args.command == 'train':
        train(model, train_data, valid_data, epochs=100)
        torch.save(model.state_dict(), model_checkpoint)
        # run evaluation in the end of training
        eval_model(model)
    elif args.command == 'eval':
        model.load_state_dict(torch.load(model_checkpoint))
        eval_model(model)
    else:
        raise ValueError(f"Unknown command: {args.command}")
