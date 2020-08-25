## Summary
This repository represents Sixthsense Project dealing with pedestrian protection solution using object detection methods, and incorporates several modules we made based on yolov5 model https://github.com/ultralytics/yolov5/blob/master/README.md

**All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.

## Code Explanation
- danger_zone.py
- detect_photo_version3.py
- makeioutable.py
- tracker.py
- ttt.py
- xyxypc2ppc.py
- detect.py
- app.py


## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.6`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab Notebook** with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)


## Inference
You can run code by running ttt.py.

But, before you run ttt.py, you might have to change ttt.py.

```
sys.stdout = open('/data/swmrepo/sunshine-2/yolov5/inference/output/output.txt', 'w') # change directory at which you want to save log of your result.
mypath = '/data/swmrepo/sunshine-2/yolov5/inference/images/DATA_DIRECTORY' # change directory at which you want to read from.
```

After you change ttt.py, you can easily run code by just running ttt.py.

```bash
$ python3 ttt.py
```

## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## Code of Conduct


## Contribution Guide

## Issue Template

## Liscence
- **GPL 3.0**


