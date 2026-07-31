import argparse
import sys
from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

root = Path(__file__).absolute().parents[1]
cls_root = Path(__file__).absolute().parent
sys.path[:0] = [str(root), str(cls_root)]

import utils.util as util
from gspointcls import GSPointCls
from utils.timm.optim import create_optimizer_v2
from utils.timm.scheduler.cosine_lr import CosineLRScheduler

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
    parser.add_argument("--run-id", default="01")
    parser.add_argument("--resume", action="store_true")
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

log_dir = dataset_dir / "output" / "log" / args.run_id
model_dir = dataset_dir / "output" / "model" / args.run_id
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
sys.stdout = open(log_dir / "out.log", "a", 1)
sys.stderr = open(log_dir / "err.log", "a", 1)

print(f"{args.dataset} base")

traindlr = DataLoader(
    dataset_class(),
    batch_size=cfg.batch_size,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    num_workers=6,
)
testdlr = DataLoader(
    dataset_class(partition="test"),
    batch_size=cfg.batch_size,
    pin_memory=True,
    persistent_workers=True,
    num_workers=6,
)

step_per_epoch = len(traindlr)

model = GSPointCls(cfg.gspoint_args).cuda()

optimizer = create_optimizer_v2(model, lr=cfg.learning_rate, weight_decay=5e-2)
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=cfg.epoch * step_per_epoch,
    lr_min=cfg.learning_rate / 10000,
    warmup_t=cfg.warmup * step_per_epoch,
    warmup_lr_init=cfg.learning_rate / 20,
)
scalar = GradScaler()

last_path = model_dir / "last.pt"
best_path = model_dir / "best.pt"
if args.resume:
    start_epoch = util.load_state(str(last_path), model=model, optimizer=optimizer, scalar=scalar)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(cfg.gspoint_args.num_classes)
ttls = util.AverageMeter()
corls = util.AverageMeter()
best = 0

for i in range(start_epoch, cfg.epoch):
    model.train()
    ttls.reset()
    corls.reset()
    metric.reset()
    now = time()
    for xyz, y in traindlr:
        lam = scheduler_step / (cfg.epoch * step_per_epoch)
        lam = 3e-3 ** lam / 3
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with autocast():
            p, closs = model(xyz)
            loss = F.cross_entropy(p, y, label_smoothing=cfg.label_smoothing)
        metric.update(p.detach(), y)
        ttls.update(loss.item())
        corls.update(closs.item())
        optimizer.zero_grad(set_to_none=True)
        loss = loss + closs * lam
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

    print(f"epoch {i}:")
    print(f"loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    metric.print("train:", iou=False)

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
    print(f"duration: {time() - now}")
    cur = metric.acc
    if best < cur:
        best = cur
        print("new best!")
        util.save_state(str(best_path), model=model)

    util.save_state(str(last_path), model=model, optimizer=optimizer, scalar=scalar, start_epoch=i + 1)
