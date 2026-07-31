# GSPoint

Official PyTorch implementation of **Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis**, AAAI 2026 (Oral).

## Repository Structure

```text
GSPointCls/
  gspointcls.py
  train.py
  test.py
  mn40/
    ModelNet40.py
    mn40_cfg.py
  sonn/
    ScanObjectNN.py
    sonn_cfg.py
GSPointSeg/
  shapenetpart/
    config.py
    gspointpartseg.py
    train.py
    test.py
  s3dis/
    config.py
    gspointsemseg.py
    prepare_s3dis.py
    train.py
    test.py
utils/
  graph_smoothing.py
  cutils/
  pointnet2_ops_lib/
  timm/
```

## Installation

The code was developed with Python 3.10, CUDA, and PyTorch 1.13.1.

```bash
conda create -n gspoint python=3.10
conda activate gspoint
```

Install a CUDA-enabled PyTorch build compatible with your system, then install dataset and point cloud dependencies:

```bash
conda install h5py
cd utils/pointnet2_ops_lib
pip install .
cd ../..
```

The CUDA/C++ operators in `utils/cutils` are compiled on first import.

## Data Preparation

### ModelNet40

Place the ModelNet40 HDF5 files under:

```text
GSPointCls/mn40/modelnet40_ply_hdf5_2048/
```

### ScanObjectNN

Set `data_path` in:

```text
GSPointCls/sonn/sonn_cfg.py
```

The expected structure is:

```text
<data_path>/main_split/training_objectdataset_augmentedrot_scale75.h5
<data_path>/main_split/test_objectdataset_augmentedrot_scale75.h5
```

### ShapeNetPart

Set `data_path` in:

```text
GSPointSeg/shapenetpart/config.py
```

For evaluation, `test.py` expects `pretrained/best.pt` by default.

### S3DIS

Set `raw_data_path` in:

```text
GSPointSeg/s3dis/config.py
```

Then preprocess the raw rooms:

```bash
cd GSPointSeg/s3dis
python prepare_s3dis.py
```

## Training and Evaluation

### Classification

ModelNet40:

```bash
cd GSPointCls
python train.py --dataset mn40 --run-id 01
python test.py --dataset mn40 --checkpoint mn40/output/model/01/best.pt
```

ScanObjectNN:

```bash
cd GSPointCls
python train.py --dataset sonn --run-id 01
python test.py --dataset sonn --checkpoint sonn/output/model/01/best.pt
```

### ShapeNetPart Part Segmentation

```bash
cd GSPointSeg/shapenetpart
python train.py
python test.py
```

### S3DIS Semantic Segmentation

```bash
cd GSPointSeg/s3dis
python train.py
python test.py
```

## Citation

```bibtex
@inproceedings{yuan2026gspoint,
  title={Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis},
  author={Yuan, Shangbo and Xu, Jie and Hu, Ping and Zhu, Xiaofeng and Zhao, Na},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
