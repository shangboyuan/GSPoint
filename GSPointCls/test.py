import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

root = Path(__file__).absolute().parents[1]
cls_root = Path(__file__).absolute().parent
sys.path[:0] = [str(root), str(cls_root)]

import utils.util as util
from gspointcls import GSPointCls

datasets = {
    "mn40": {
        "dir": "mn40",
        "module": "ModelNet40",
        "class": "ModelNet40",
        "cfg": "mn40_cfg",
    },
    "sonn": {
        "dir": "sonn",
        "module": "ScanObjectNN",
        "class": "ScanObjectNN",
        "cfg": "sonn_cfg",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(datasets), default="mn40")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def load_components(dataset):
    item = datasets[dataset]
    dataset_dir = cls_root / item["dir"]
    sys.path.insert(0, str(dataset_dir))
    data_module = __import__(item["module"], fromlist=[item["class"]])
    cfg = __import__(item["cfg"])
    dataset_class = getattr(data_module, item["class"])
    return dataset_dir, cfg, dataset_class


torch.set_float32_matmul_precision("high")

args = parse_args()
dataset_dir, cfg, dataset_class = load_components(args.dataset)
checkpoint = Path(args.checkpoint) if args.checkpoint else dataset_dir / "pretrained" / "best.pt"

testdlr = DataLoader(
    dataset_class(partition="test"),
    batch_size=cfg.batch_size,
    pin_memory=True,
    num_workers=6,
)

model = GSPointCls(cfg.gspoint_args).cuda()
util.load_state(str(checkpoint), model=model)

metric = util.Metric(cfg.gspoint_args.num_classes)

model.eval()
metric.reset()
with torch.no_grad():
    for xyz, y in testdlr:
        xyz = xyz.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with autocast():
            p = model(xyz)
        metric.update(p, y)

metric.print("val:  ", iou=False)
