# utils.py

import torch
from tqdm import tqdm

criterion = torch.nn.CrossEntropyLoss()

def train_epoch(model, device, train_loader, optimizer, epoch, train_losses=[], train_acc=[]):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    correct, processed = 0, 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        processed += len(data)
        acc = 100 * correct / processed
        train_acc.append(acc)

        pbar.set_description(f"Loss={loss.item():.4f} Batch={batch_idx} Acc={acc:.2f}%")

def test_model(model, device, test_loader, test_losses=[], test_acc=[]):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    acc = 100. * correct / len(test_loader.dataset)
    test_acc.append(acc)
    print(f"\nTest set: Avg Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")
