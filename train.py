import numpy as np
import os
import torch
import torch.optim as optim
import yaml

from dataloader import get_dataloader
from efficient_kan import KAN
from evaluate import evaluate_model
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

def main():
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    train_csv = "data/train.csv"
    val_csv = "data/val.csv"
    save_path = "model.pkl"
    best_save_path = "best_model_7503.pkl"

    num_classes = config['num_classes']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    input_dim = config['num_features']
    num_output = config['num_output']
    layers_hidden = config['layers_hidden']
    epochs = config['epochs']
    lr = config['learning_rate']
    use_well = config['use_WELL']
    use_depth = config['use_DEPTH']
    update_grid_interval = config['update_grid_interval']


    print("\nLoading training data...")
    train_loader, train_labels = get_dataloader(
        train_csv,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        use_depth=use_depth,
        use_well=use_well
    )

    print("\nLoading validation data...")
    val_loader, _ = get_dataloader(
        val_csv,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        use_depth=use_depth,
        use_well=use_well
    )


    input_dim = len(train_loader.dataset.features[0])

    # 定义网络结构：[input_dim, hidden1, hidden2, ..., num_classes]
    # layers_hidden = [input_dim, 128, 64, 32, num_classes]

    model = KAN(
        layers_hidden=layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_base=2.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_range=[-2, 2],
        device=device
    ).to(device)

    class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


    best_acc = 0.0
    start_epoch = 0
    print(f"\n\nStart training on {device}...")

    if os.path.exists(best_save_path):
        model.load_state_dict(torch.load(best_save_path), strict=False)
        print(f"Model loaded from {best_save_path}")

    for epoch in range(start_epoch, start_epoch + epochs):
        # model.train()
        running_loss = 0.0
        i = 0

        for i, (x, y, z) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y) + model.regularization_loss()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{start_epoch + epochs}], Step [{i}/{(len(train_loader.dataset)/batch_size):.0f}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        print(f"\n 正在评估第 {epoch + 1} 轮...")
        results = evaluate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            # save_dir=f"results/epoch_{epoch + 1}",  # 每轮保存一次结果
            save_dir=None,
            dataset=val_loader.dataset
        )

        val_acc = results['accuracy']

        print(f"\nEpoch [{epoch + 1}/{start_epoch + epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("\n>>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_save_path)
            print(f"Best model saved with accuracy: {best_acc:.4f}")
        
        torch.save(model.state_dict(), save_path)

        # --- 调整学习率 ---
        # scheduler.step(val_acc)
        # print(f"\nCurrent learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # --- 定期更新 grid ---
        if (epoch + 1) % update_grid_interval == 0:
            print(f"\nUpdating KAN grids at epoch {epoch+1}")
            model.eval()
            with torch.no_grad():
                sample_inputs, _, _ = next(iter(train_loader))
                sample_inputs = sample_inputs.to(device)
                model(sample_inputs, update_grid=True)

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()