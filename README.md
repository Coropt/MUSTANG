# MUSTANG

This repository is currently being organized and cleaned for review purposes. The full, production-ready version with detailed documentation will be released upon acceptance.

## Repository Architecture

```text
CSDI/
├── main_conditional_diffusion.py
├── train_mustang.py
├── config/
│   └── base.yaml
├── models/
│   ├── conditional_diffusion.py
│   ├── graph_aware_conditional_diffusion.py
│   ├── diff_models.py
│   └── diff_models_graph.py
├── util/
│   ├── utils.py
│   └── meta_learning.py
├── scripts/
│   ├── run_mustang.sh
│   └── run_mustang_pretrained_mask.sh
└── dataset_process/
    ├── dataset_discharge.py
    ├── dataset_pooled.py
    ├── dataset_no3.py
    ├── dataset_nh4.py
    ├── dataset_srp.py
    ├── dataset_tp.py
    ├── dataset_ssc.py
    ├── meta_data_processor.py
    └── utils.py
```

## Entry Points

- `main_conditional_diffusion.py`: single-domain conditional diffusion training/evaluation.
- `train_meta_csdi.py`: backward-compatible launcher that forwards to `train_mustang.py`.
- `train_mustang.py`: MUSTANG meta-training pipeline.
- `visualize_conditional_diffusion_predictions.py`: prediction visualization utility.

## Example Scripts

- `scripts/run_mustang.sh`: meta-training followed by fine-tuning.
- `scripts/run_mustang_pretrained_mask.sh`: meta-training + fine-tuning with pretrained masking.
