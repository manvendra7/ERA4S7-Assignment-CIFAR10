## train.py

import torch
from torch import optim
from model_1 import Model_1
from dataloader import get_loaders
from utils import train_epoch, test_model
from torch.optim.lr_scheduler import StepLR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device -->", device)

    train_loader, test_loader = get_loaders(batch_size=64)

    model = Model_1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)  # reduce LR by 10x every 15 epochs

    EPOCHS = 50
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []

    for epoch in range(1, EPOCHS + 1):
        print("EPOCH:", epoch)
        train_epoch(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
        test_model(model, device, test_loader, test_losses, test_acc)
        scheduler.step()

if __name__ == "__main__":
    main()