from types import SimpleNamespace
from pathlib import Path
from torch import nn
import torch

data_path = Path("xxx/shapenetcore_partanno_segmentation_benchmark_v0_normal")

presample_path = data_path.parent / "shapenet_part_presample.pt"

epoch = 250
warmup = 20
batch_size = 32
learning_rate = 2e-3
label_smoothing = 0.2

gspoint_args = SimpleNamespace()
gspoint_args.depths = [4, 4, 4, 4]
gspoint_args.ns = [2048, 512, 192, 64]
gspoint_args.ks = [20, 20, 20, 20]
gspoint_args.dims = [96, 192, 320, 512]
gspoint_args.nbr_dims = [48,48]
gspoint_args.head_dim = 320
gspoint_args.num_classes = 50
gspoint_args.shape_classes = 16
gspoint_args.graph_smoothing_steps = 3
gspoint_args.graph_smoothing_alpha = 0.5
gspoint_args.eigen_dim = 32
drop_path = 0.15
drop_rates = torch.linspace(0., drop_path, sum(gspoint_args.depths)).split(gspoint_args.depths)
gspoint_args.drop_paths = [dpr.tolist() for dpr in drop_rates]
gspoint_args.head_drops = torch.linspace(0., 0.15, len(gspoint_args.depths)).tolist()
gspoint_args.bn_momentum = 0.1
gspoint_args.act = nn.GELU
gspoint_args.mlp_ratio = 2
gspoint_args.cor_std = [0.75, 1.5, 2.5, 4.7]
