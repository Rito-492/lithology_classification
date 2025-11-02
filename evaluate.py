
import numpy as np
import os
import yaml
import torch
import pandas as pd

from efficient_kan import *
from dataloader import get_dataloader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, val_loader, device, save_dir=None, dataset=None):
    """
    完整评估模型性能
    Args:
        model: 训练好的KAN模型
        val_loader: 验证集DataLoader
        device: 运行设备 ('cuda' or 'cpu')
        save_dir (str): 如果指定，则保存评估结果到该目录
        dataset: 可选，用于提取id和原始数据（如果需要输出预测详情）

    Returns:
        results (dict): 包含 accuracy, report, cm 等信息
    """
    # model.eval()
    all_preds = []
    all_labels = []
    all_features = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 2:
                x, y = batch
                ids = None
            elif len(batch) == 3:
                x, y, ids = batch
            else:
                raise ValueError("DataLoader should return (x, y) or (x, y, ids)")

            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            if ids is not None:
                all_ids.extend(ids.numpy())
            all_features.append(x.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_features = np.concatenate(all_features, axis=0) if all_features else np.array([])

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['粉砂岩', '砂岩', '泥岩'],
        digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': acc,
        'report_str': report,
        'classification_report': classification_report(
            all_labels, all_preds, target_names=['粉砂岩', '砂岩', '泥岩'], output_dict=True
        ),
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'ids': all_ids
    }

    print(f" Validation Accuracy: {acc * 100:.2f}%")
    print("\n 分类报告:")
    print(report)
    print("\n 混淆矩阵:")
    print(cm)

    return results

if __name__ == '__main__':

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = config['num_features']
    num_classes = config['num_classes']

    val_loader, _ = get_dataloader(
        csv_path='data/val.csv',
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        use_depth=config['use_DEPTH'],
        use_well=config['use_WELL']
    )

    layers_hidden = config['layers_hidden']

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

    if os.path.exists('best_model.pkl'):
        model.load_state_dict(torch.load('best_model.pkl', map_location=device))

    print(f"\nStart evaluating on {device}...")

    print("\n>>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    accuracy = evaluate_model(
        model,
        val_loader,
        device,
        save_dir='result/all',
        dataset=val_loader.dataset
    )['accuracy']
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')