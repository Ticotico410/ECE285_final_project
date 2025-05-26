# ECE285 Final Project
# Author: Chenbin Yu, Riqian Hu

## Overview

This repository contains a command-line script (`main.py`) for end-to-end training and evaluation of an enhanced YOLOv8 model on custom datasets (e.g., VisDrone). It automates experiment management by organizing checkpoints, detection outputs, metrics, and logs into a consistent `runs/<experiment>` directory.

---

## Requirements

- Python 3.8+
- [ultralytics](https://github.com/ultralytics/ultralytics) (installed via `requirements.txt`)
- Other dependencies listed in `requirements.txt`.

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## File Structure

```
├── main.py                # CLI entry point for training/testing
├── requirements.txt       # Python dependencies
├── setup.py               # Ultralytics dependencies
├── ultralytics/           # Modified YOLOv8 source code and configs
│   ├── cfg/               # Model configuration files
│   ├── data/              # Dataset loaders and scripts
│   ├── engine/            # Training/validation engine
│   ├── models/            # YOLOv8 model definitions
│   ├── nn/                # YOLOv8 modules based on PyTorch
│   └── utils/             # Evaluation metrics
└── runs/                  # Generated output directories (checkpoints, logs, results)
```

---

## Usage

### Command-Line Arguments

| Option      | Description                                                                                                 | Default                                                             |
|-------------|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| `--mode`    | Operation mode: `train` or `test`                                                                           | **Required**                                                        |
| `--model`   | Path to YOLOv8 weights or model name (e.g., `yolov8l.pt`)                                                    | `runs/yolov8l/train/weights/yolov8l_best.pt`                        |

### Running Training

To train the model on your dataset (specified in `myVisDrone.yaml` or other YAML config):

```bash
python main.py --mode train --model runs/yolov8l/train/weights/yolov8l_best.pt
```

- **Output**: Checkpoints, training logs and metrics will be saved under:

```
runs/yolov8l/train/
```

### Running Evaluation / Testing

To evaluate on the validation set and generate detection results:

```bash
python main.py --mode test --model runs/yolov8l/train/weights/yolov8l_best.pt
```

- **Output**:
  - Detection results (TXT, JSON) and evaluation metrics (mAP, precision, recall) under:

```
runs/yolov8l/val/
```

  - A `myresults.txt` file summarizing key metrics and timing breakdown.
  - The `test.log` file automatically copied into the latest validation folder for easy reference.

---

## Experiment Management

- The script infers an experiment name from the `--model` path (e.g., `yolov8l` from `runs/yolov8l/...`).
- All outputs (train/val) are organized under `runs/<experiment>` for consistency and reproducibility.

---

## Customization

- Edit or add dataset configurations in the `.yaml` files under `ultralytics/cfg/models/` or your own YAML in the root.
- Tune training hyperparameters (epochs, batch size, learning rate) by modifying the `model.train(...)` call in `main.py`.

---

## License & Citation

Please cite this work if you use it in your research or projects.
