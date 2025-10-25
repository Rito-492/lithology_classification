# 基于测井曲线的岩性识别与分类

## 2025-10-21 7503
1. config:
    num_classes: 3
    class_names: ['1', '2', '3']
    batch_size: 128
    num_workers: 8
    num_features: 4
    num_hidden: 3
    num_outputs: 3
    epochs: 100
    update_grid_interval: 50
    learning_rate: 0.0001
    use_WELL: false
    use_DEPTH: true
2. best_model: best_model_7503.pkl
3. layers_hidden = [input_dim, 128, 64, 32, num_classes]
4. model = KAN(
        layers_hidden=layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_base=2.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_range=[-2, 2],
        device=device
    ).to(device)