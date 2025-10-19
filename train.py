# -*- coding: utf-8 -*-

import os
import torch
import torch.optim as optim
import yaml

from dataloader import get_dataloader
from efficient_kan import KAN
from evaluate import evaluate_model
from torch import nn

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_csv = "data/train.csv"
val_csv = "data/validation.csv"
save_path = "model.pkl"
best_save_path = "best_model.pkl"

num_classes = config['num_classes']
batch_size = config['batch_size']
epochs = config['epochs']
lr = config['learning_rate']
use_well = config['use_WELL']
use_depth = config['use_DEPTH']


print("Loading training data...")
train_loader = get_dataloader(
    train_csv,
    batch_size=batch_size,
    shuffle=True,
    use_depth=use_depth,
    use_well=use_well
)
print(f"Classes: {num_classes}")

print("Loading validation data...")
val_loader = get_dataloader(
    val_csv,
    batch_size=batch_size,
    shuffle=False,
    use_depth=use_depth,
    use_well=use_well
)


input_dim = len(train_loader.dataset.features[0])

# 定义网络结构：[input_dim, hidden1, hidden2, ..., num_classes]
layers_hidden = [input_dim, 64, 32, num_classes]

model = KAN(
    layers_hidden=layers_hidden,
    grid_size=5,
    spline_order=3,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation=torch.nn.SiLU,
    grid_range=[-1, 1],
    device=device
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


best_acc = 0.0
print(f"Start training on {device}...")

if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path)).to(device)
    print(f"Model loaded from {save_path}")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y) + model.regularization_loss()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"\n 正在评估第 {epoch + 1} 轮...")
    results = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        save_dir=f"results/epoch_{epoch + 1}",  # 每轮保存一次结果
        dataset=val_loader.dataset
    )

    val_acc = results['accuracy']

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_save_path)
        print(f"Best model saved with accuracy: {best_acc:.4f}")
    
    torch.save(model.state_dict(), save_path)

print(f"Training finished. Best validation accuracy: {best_acc:.4f}")