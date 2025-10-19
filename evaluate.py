
import os
import yaml
import torch
import pandas as pd

from efficient_kan import *
from dataloader import get_dataloader


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == '__main__':

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = config['num_features']
    num_classes = config['num_classes']

    val_loader = get_dataloader(
        csv_path='data/val.csv',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_depth=config['use_DEPTH'],
        use_well=config['use_WELL']
    )

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

    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))

    accuracy = evaluate_model(model, val_loader, device)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')