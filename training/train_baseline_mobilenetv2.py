"""
Training Baseline MobileNetV2 on CIFAR-10
baseline_mobilenet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy


from models.model_mobilenetv2 import get_mobilenetv2
from dataset.cifar_10 import load_cifar10
from utils.seed_everything import seed_everything
from utils.logger import logger
from utils.metrics import accuracy

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

train_dataset, test_dataset = load_cifar10("./data")

from torch.utils.data import random_split, DataLoader
train_len = int(0.9 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_data, val_data = random_split(train_dataset, [train_len, val_len])

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# model, optimizer and schedulers
model = get_mobilenetv2(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            total += y.size(0)
            correct += preds.eq(y).sum().item()
    return 100.0 * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss / len(train_loader), acc=100. * correct / total)

        scheduler.step()
        val_acc = evaluate(model, val_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    logger.info(f"Best Validation Accuracy: {best_acc:.2f}%")
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50)
    test_acc = evaluate(model, test_loader)
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), "./models/mobilenetv2_cifar10_best.pth")
    logger.info("Model saved as mobilenetv2_cifar10_best.pth")
