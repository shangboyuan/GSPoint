from types import SimpleNamespace
from torch import nn
import torch
from pathlib import Path

data_path = Path("xxx/")

epoch = 250
warmup = 20
batch_size = 32
learning_rate = 3e-3
label_smoothing = 0.2

gspoint_args = SimpleNamespace()
gspoint_args.depths = [4, 4, 4, 4]
gspoint_args.ns = [1024, 256, 64, 32]
gspoint_args.ks = [24, 24, 24, 16]
gspoint_args.dims = [96, 192, 384, 512]
gspoint_args.nbr_dims = [48, 48]
gspoint_args.bottleneck = 2048
gspoint_args.num_classes = 15
gspoint_args.num_points = 1024
gspoint_args.sample_train = True
gspoint_args.sample_eval = True
gspoint_args.train_sample_points = 1200
gspoint_args.height_scale = 40
gspoint_args.height_jitter = 0.0
gspoint_args.graph_smoothing_steps = 3
gspoint_args.graph_smoothing_alpha = 0.5
gspoint_args.eigen_dim = 32
drop_path = 0.1
drop_rates = torch.linspace(0., drop_path, sum(gspoint_args.depths)).split(gspoint_args.depths)
gspoint_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
gspoint_args.bn_momentum = 0.1
gspoint_args.act = nn.GELU
gspoint_args.mlp_ratio = 2
gspoint_args.cor_std = [2.2, 4.4, 8.8, 17.6]
