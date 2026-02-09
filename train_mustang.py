import argparse
import datetime
import json
import os
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
import yaml

from dataset_process.dataset_discharge import get_dataloader_discharge, Discharge_Dataset
from dataset_process.dataset_no3 import get_dataloader_NO3, NO3_Dataset
from dataset_process.dataset_nh4 import get_dataloader_NH4, NH4_Dataset
from dataset_process.dataset_srp import get_dataloader_SRP, SRP_Dataset
from dataset_process.dataset_tp import get_dataloader_TP, TP_Dataset
from dataset_process.dataset_ssc import get_dataloader_SSC, SSC_Dataset
from dataset_process.meta_data_processor import build_nonoverlap_loaders
from models.conditional_diffusion import ConditionalDiffusionImputation
from models.graph_aware_conditional_diffusion import GraphAwareConditionalDiffusionImputation
from util.meta_learning import (
    MetaTaskLoader, 
    MultiDatasetMetaTaskLoader,
    MultiDatasetMetaTaskLoaderV2,
    MultiDatasetMetaTaskLoaderV3,
    RoundRobinMetaTaskLoader,
    meta_train, 
    meta_train_v2,
    meta_train_v3,
    meta_validate,
    meta_validate_v2,
    meta_validate_v3,
)

COMMON_STATIONS_12 = [
    "04178000",
    "04182000",
    "04183000",
    "04183500",
    "04185318",
    "04186500",
    "04188100",
    "04190000",
    "04191058",
    "04191444",
    "04191500",
    "04192500",
]


def _str2bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "t"}


def _resolve_target_dim(args: argparse.Namespace) -> int:
    if args.target_dim is not None:
        return args.target_dim
    # All datasets now use the common station set defined above.
    if args.dataset in ("discharge", "nh4", "no3", "srp", "tp", "ssc"):
        return len(COMMON_STATIONS_12)
    raise ValueError("Please provide --target_dim for custom datasets.")


def _get_dataloader_fn(dataset: str) -> Callable:
    """Get the appropriate dataloader function for the given dataset."""
    loaders = {
        "discharge": get_dataloader_discharge,
        "nh4": get_dataloader_NH4,
        "no3": get_dataloader_NO3,
        "srp": get_dataloader_SRP,
        "tp": get_dataloader_TP,
        "ssc": get_dataloader_SSC,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
    return loaders[dataset]


def _get_dataset_class(dataset: str):
    """Get the appropriate Dataset class for the given dataset."""
    classes = {
        "discharge": Discharge_Dataset,
        "nh4": NH4_Dataset,
        "no3": NO3_Dataset,
        "srp": SRP_Dataset,
        "tp": TP_Dataset,
        "ssc": SSC_Dataset,
    }
    if dataset not in classes:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(classes.keys())}")
    return classes[dataset]


def get_data_loaders(
    dataset: str,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    training_missing: str = "point",
    val_len: float = 0.1,
    test_len: float = 0.2,
    seed: int = 1,
    meta_batch_sampler: str = "nonoverlap",
) -> Tuple:
    """
    Get data loaders for the specified dataset.
    
    The new MUSTANG approach:
      - Each TASK = time-based split of ALL stations together
      - Support set: some time points (all stations) - used for inner loop conditional diffusion training
      - Query set: remaining time points (all stations) - used for meta-objective
      - This preserves conditional diffusion's multi-station joint modeling!
    
    Args:
        dataset: Name of the dataset ('discharge', 'tp', etc.)
        sequence_length: Length of each sequence
        batch_size: Batch size
        device: PyTorch device
        training_missing: Missing pattern ('point' or 'block')
        seed: Random seed
    Returns:
        Tuple of (train_loader, valid_loader, target_dim)
    """
    # All datasets now use the common station set defined above.
    target_dim = len(COMMON_STATIONS_12)
    
    if meta_batch_sampler == "nonoverlap":
        DatasetClass = _get_dataset_class(dataset)
        train_loader, valid_loader = build_nonoverlap_loaders(
            DatasetClass,
            sequence_length=sequence_length,
            batch_size=batch_size,
            seed=seed,
            training_missing=training_missing,
            val_len=val_len,
            test_len=test_len,
            num_workers=4,
        )
        print("  Using non-overlapping batch sampler for MUSTANG")
    else:
        # Use the default dataloader functions (shuffled)
        get_dataloader = _get_dataloader_fn(dataset)
        train_loader, valid_loader, _, _, _ = get_dataloader(
            sequence_length=sequence_length,
            device=device,
            training_missing=training_missing,
            test_missing="point",
            seed=seed,
            batch_size=batch_size,
            val_len=val_len,
            test_len=test_len,
            missing_ratio=0.1,
        )
    
    return train_loader, valid_loader, target_dim


def _normalize_station_id(value) -> str:
    try:
        return f"{int(value):08d}"
    except (TypeError, ValueError):
        return str(value)


def _load_flow_direction(path: str, station_ids):
    df = pd.read_csv(path, index_col=0)
    df.index = [_normalize_station_id(x) for x in df.index]
    df.columns = [_normalize_station_id(x) for x in df.columns]
    missing = [s for s in station_ids if s not in df.index or s not in df.columns]
    if missing:
        raise ValueError(f"Adjacency matrix missing stations: {missing}")
    return df.loc[station_ids, station_ids].values


def _load_meta_config(config: Dict) -> Dict:
    """Load MUSTANG configuration with sensible defaults."""
    default_meta = {
        "tasks_per_batch": 4,
        "support_frac": 0.5,
        "inner_steps": 3,
        "inner_epochs": 1,  # Number of times to iterate through support batches
        "inner_lr": 1.0e-3,
        "num_outer_steps": 500,
        "grad_clip": 1.0,
        "log_interval": 10,
        "val_interval": 20,  # Validate every N outer steps
        "patience": 10,  # Early stopping patience (number of validations without improvement)
    }
    user_meta = config.get("meta", {})
    return {**default_meta, **user_meta}


def main():
    parser = argparse.ArgumentParser(description="MUSTANG for conditional diffusion (MAML-style)")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dataset",
        type=str,
        default="discharge",
        choices=["discharge", "nh4", "no3", "srp", "tp"],
        help="Single dataset for MUSTANG (ignored if --datasets is used)."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Multiple datasets for cross-dataset MUSTANG (e.g., --datasets tp no3 nh4 srp)."
    )
    parser.add_argument(
        "--held_out",
        type=str,
        default=None,
        help="Dataset to exclude from meta-training (for leave-one-out evaluation)."
    )
    parser.add_argument("--target_dim", type=int, default=None, 
                        help="Override variable count for custom datasets.")
    parser.add_argument("--sequence_length", type=int, default=16, 
                        help="Length of each sampled sequence.")
    parser.add_argument("--save_folder", type=str, default="", 
                        help="Optional directory to store meta-trained weights.")
    parser.add_argument("--training_missing", type=str, default="point",
                        choices=["point", "block"],
                        help="Missing pattern for training.")
    parser.add_argument("--val_len", type=float, default=0.1,
                        help="Fraction of data used for validation split in MUSTANG.")
    parser.add_argument("--test_len", type=float, default=0.2,
                        help="Fraction of data reserved for test split in MUSTANG.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--round_robin", action="store_true",
                        help="Use round-robin task sampling (one batch from each dataset per epoch) instead of random sampling.")
    parser.add_argument(
        "--meta_batch_sampler",
        type=str,
        choices=["nonoverlap", "default"],
        default="nonoverlap",
        help="Batch sampling strategy for MUSTANG loaders.",
    )
    parser.add_argument("--v3", action="store_true",
                        help="Use V3 MUSTANG: mask-based support/query split (preserves shape).")
    parser.add_argument("--use_wandb", type=_str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="mustang-conditional-diffusion")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--use_gcn", action="store_true",
                        help="Use graph-aware conditional diffusion spatial layer for MUSTANG.")
    parser.add_argument("--use_graphdiffusion", action="store_true",
                        help="Use CAPRI-style graph-aware conditional diffusion spatial layer for MUSTANG.")
    parser.add_argument("--adj_path", type=str, default="",
                        help="Path to adjacency/flow-direction CSV (defaults to Discharge SSC_sites_flow_direction.csv).")
    parser.add_argument("--num_outer_steps", type=int, default=None,
                        help="Override MUSTANG outer steps (meta-training 'epochs'). Default from config meta.num_outer_steps.")
    args = parser.parse_args()

    if args.val_len < 0 or args.test_len < 0 or args.val_len + args.test_len >= 1:
        raise ValueError("val_len and test_len must be >= 0 and sum to < 1.")
    
    # Determine which datasets to use
    all_datasets = ["discharge", "tp", "no3", "nh4", "srp", "ssc"]
    if args.datasets:
        train_datasets = args.datasets
    elif args.held_out:
        train_datasets = [d for d in all_datasets if d != args.held_out]
    else:
        train_datasets = [args.dataset]
    
    multi_dataset_mode = len(train_datasets) > 1
    
    print(f"=== MUSTANG Conditional Diffusion (First-Order MAML) ===")
    print(f"Mode: {'Multi-Dataset' if multi_dataset_mode else 'Single-Dataset'}")
    print(f"Training datasets: {train_datasets}")
    if args.held_out:
        print(f"Held-out dataset: {args.held_out}")
    print(f"Device: {args.device}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Task sampling: {'Round-Robin' if args.round_robin else 'Random'}")

    device = torch.device(args.device)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join("config", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.use_gcn and args.use_graphdiffusion:
        raise ValueError("Choose only one of --use_gcn or --use_graphdiffusion.")
    graph_model = None
    if args.use_graphdiffusion:
        graph_model = "graphdiffusion"
    elif args.use_gcn:
        graph_model = "graphormer"
    if graph_model:
        config["diffusion"]["graph_model"] = graph_model

    meta_config = _load_meta_config(config)
    if args.num_outer_steps is not None:
        meta_config["num_outer_steps"] = args.num_outer_steps
        print(f"Overriding num_outer_steps to {args.num_outer_steps}")
    target_dim = _resolve_target_dim(args)

    print(f"Target dimension: {target_dim}")
    print(f"Meta config: {json.dumps(meta_config, indent=2)}")

    # Build model
    adj = None
    adj_path = None
    if graph_model:
        adj_path = args.adj_path or os.path.join(
            "original_data", "Discharge", "SSC_sites_flow_direction.csv"
        )
        adj = _load_flow_direction(adj_path, COMMON_STATIONS_12)
        print(f"Using graph-aware conditional diffusion adjacency from: {adj_path}")

    if graph_model:
        model = GraphAwareConditionalDiffusionImputation(
            config, args.device, target_dim=target_dim, adj=adj
        ).to(device)
    else:
        model = ConditionalDiffusionImputation(config, args.device, target_dim=target_dim).to(device)
    
    # Optimizer with weight decay
    # Use meta.outer_lr if available, otherwise fall back to train.lr
    outer_lr = meta_config.get("outer_lr", config["train"]["lr"])
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=outer_lr, 
        weight_decay=1e-6
    )
    
    # Fixed learning rate (no scheduler)
    print(f"Using fixed learning rate (outer loop): {outer_lr}")

    # Load data
    print("\nLoading data...")
    
    if multi_dataset_mode:
        # Multi-dataset mode: load all training datasets
        train_loaders = {}
        valid_loaders = {}
        
        for dataset_name in train_datasets:
            print(f"  Loading {dataset_name}...")
            train_loader, valid_loader, _ = get_data_loaders(
                dataset=dataset_name,
                sequence_length=args.sequence_length,
                batch_size=config["train"]["batch_size"],
                device=device,
                training_missing=args.training_missing,
                val_len=args.val_len,
                test_len=args.test_len,
                seed=args.seed,
                meta_batch_sampler=args.meta_batch_sampler,
            )
            train_loaders[dataset_name] = train_loader
            valid_loaders[dataset_name] = valid_loader
        
        print(f"\nLoaded {len(train_datasets)} datasets for multi-dataset MUSTANG")
        
        # Determine which version to use
        use_v2 = False
        use_v3 = args.v3
        
        # Create multi-dataset meta-loaders
        if use_v3:
            # V3: Mask-based split (preserves shape) - only random sampling
            print("Using MultiDatasetMetaTaskLoaderV3: mask-based split with random sampling")
            print(f"  Support fraction: {meta_config['support_frac']}")
            print(f"  Inner steps: {meta_config['inner_steps']}")
            train_meta_loader = MultiDatasetMetaTaskLoaderV3(
                data_loaders=train_loaders,
                support_frac=meta_config["support_frac"],
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
            val_meta_loader = MultiDatasetMetaTaskLoaderV3(
                data_loaders=valid_loaders,
                support_frac=meta_config["support_frac"],
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
        elif args.round_robin:
            # V1: Round-robin uses time-based splitting within batches
            print("Using RoundRobinMetaTaskLoader (V1): one task from each dataset per epoch")
            train_meta_loader = RoundRobinMetaTaskLoader(
                data_loaders=train_loaders,
                support_frac=meta_config["support_frac"],
            )
            
            val_meta_loader = RoundRobinMetaTaskLoader(
                data_loaders=valid_loaders,
                support_frac=meta_config["support_frac"],
            )
        else:
            # V2: Each inner step uses a DIFFERENT full batch
            query_batches = meta_config.get("query_batches", 16)
            inner_epochs = meta_config.get("inner_epochs", 1)
            print("Using MultiDatasetMetaTaskLoaderV2: random task sampling with full batches")
            print(f"  Each task uses {meta_config['inner_steps'] + query_batches} batches "
                  f"({meta_config['inner_steps']} support + {query_batches} query)")
            print(f"  Inner loop: {meta_config['inner_steps']} batches × {inner_epochs} epochs = "
                  f"{meta_config['inner_steps'] * inner_epochs} training steps")
            train_meta_loader = MultiDatasetMetaTaskLoaderV2(
                data_loaders=train_loaders,
                inner_steps=meta_config["inner_steps"],
                query_batches=query_batches,
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
            
            val_meta_loader = MultiDatasetMetaTaskLoaderV2(
                data_loaders=valid_loaders,
                inner_steps=meta_config["inner_steps"],
                query_batches=query_batches,
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
            use_v2 = True
    else:
        # Single dataset mode
        train_loader, valid_loader, data_target_dim = get_data_loaders(
            dataset=train_datasets[0],
            sequence_length=args.sequence_length,
            batch_size=config["train"]["batch_size"],
            device=device,
            training_missing=args.training_missing,
            val_len=args.val_len,
            test_len=args.test_len,
            seed=args.seed,
            meta_batch_sampler=args.meta_batch_sampler,
        )
        
        print(f"Data loaded. Target dim from data: {data_target_dim}")

        use_v2 = False
        use_v3 = args.v3
        
        if use_v3:
            # V3: Mask-based split for single dataset
            print("Using MultiDatasetMetaTaskLoaderV3 for single dataset: mask-based split")
            print(f"  Support fraction: {meta_config['support_frac']}")
            print(f"  Inner steps: {meta_config['inner_steps']}")
            train_meta_loader = MultiDatasetMetaTaskLoaderV3(
                data_loaders={train_datasets[0]: train_loader},
                support_frac=meta_config["support_frac"],
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
            val_meta_loader = MultiDatasetMetaTaskLoaderV3(
                data_loaders={train_datasets[0]: valid_loader},
                support_frac=meta_config["support_frac"],
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
        else:
            # V2: Each inner step uses a different full batch
            query_batches = meta_config.get("query_batches", 16)
            inner_epochs = meta_config.get("inner_epochs", 1)
            print("Using MultiDatasetMetaTaskLoaderV2 for single dataset: full batches")
            print(f"  Each task uses {meta_config['inner_steps'] + query_batches} batches "
                  f"({meta_config['inner_steps']} support + {query_batches} query)")
            print(f"  Inner loop: {meta_config['inner_steps']} batches × {inner_epochs} epochs = "
                  f"{meta_config['inner_steps'] * inner_epochs} training steps")
            train_meta_loader = MultiDatasetMetaTaskLoaderV2(
                data_loaders={train_datasets[0]: train_loader},
                inner_steps=meta_config["inner_steps"],
                query_batches=query_batches,
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
            
            val_meta_loader = MultiDatasetMetaTaskLoaderV2(
                data_loaders={train_datasets[0]: valid_loader},
                inner_steps=meta_config["inner_steps"],
                query_batches=query_batches,
                tasks_per_batch=meta_config["tasks_per_batch"],
            )
            use_v2 = True

    # Prepare save folder
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_folder:
        foldername = args.save_folder
    elif multi_dataset_mode:
        datasets_str = "-".join(sorted(train_datasets))
        task_type = "roundrobin" if args.round_robin else "random"
        if args.held_out:
            foldername = f"./save/meta/multi_excl_{args.held_out}_{task_type}/MUSTANG-{current_time}/"
        else:
            foldername = f"./save/meta/multi_{datasets_str}_{task_type}/MUSTANG-{current_time}/"
    else:
        foldername = f"./save/meta/{train_datasets[0]}/MUSTANG-{current_time}/"
    os.makedirs(foldername, exist_ok=True)
    
    # Save config
    full_config = {
        "model_config": config,
        "meta_config": meta_config,
        "args": vars(args),
        "train_datasets": train_datasets,
        "multi_dataset_mode": multi_dataset_mode,
        "held_out": args.held_out,
        "round_robin": args.round_robin,
        "use_v2": use_v2,
        "use_v3": use_v3,
        "graph_model": graph_model,
        "adj_path": args.adj_path,
    }
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(full_config, f, indent=4)

    wandb_run = None
    if args.use_wandb:
        import wandb

        run_name = args.wandb_name or f"meta_{current_time}"
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": run_name,
            "config": {
                "stage": "meta_train",
                "model_config": config.get("model", {}),
                "train_config": config.get("train", {}),
                "train_datasets": train_datasets,
                "held_out": args.held_out,
                "seed": args.seed,
                "sequence_length": args.sequence_length,
                "training_missing": args.training_missing,
                "batch_size": config["train"]["batch_size"],
                "lr": config["train"]["lr"],
                "meta_config": meta_config,
                "version": "v3" if use_v3 else ("v2" if use_v2 else "v1"),
                "model_variant": "graph-aware conditional diffusion" if graph_model else "conditional diffusion",
                "adj_path": adj_path,
                "args": vars(args),
            },
            "tags": ["meta-train"] + train_datasets,
        }
        if args.wandb_group:
            wandb_kwargs["group"] = args.wandb_group
        if args.wandb_tags:
            wandb_kwargs["tags"].extend(
                [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
            )
        if args.wandb_mode:
            wandb_kwargs["mode"] = args.wandb_mode
        wandb_run = wandb.init(**wandb_kwargs)

    # Determine version label
    if use_v3:
        version_label = "V3 (mask-based split)"
    elif use_v2:
        version_label = "V2 (full batches)"
    else:
        version_label = "V1 (time-split)"

    print(f"\n=== Starting MUSTANG Training ===")
    print(f"Output folder: {foldername}")
    print(f"Using {version_label} MUSTANG")
    
    # Meta-training with periodic validation and early stopping
    val_interval = meta_config.get("val_interval", 20)
    patience = meta_config.get("patience", 10)
    val_tasks = meta_config.get("val_tasks", 10)
    best_model_path = os.path.join(foldername, "best_meta_model.pth")
    
    # Run meta-training (First-Order MAML)
    if use_v3:
        # V3: Mask-based split (preserves shape) with early stopping
        meta_losses = meta_train_v3(
            model=model,
            meta_loader=train_meta_loader,
            optimizer=optimizer,
            inner_steps=meta_config["inner_steps"],
            inner_lr=meta_config["inner_lr"],
            num_outer_steps=meta_config["num_outer_steps"],
            grad_clip=meta_config["grad_clip"],
            scheduler=None,
            log_interval=meta_config["log_interval"],
            wandb_run=wandb_run,
            # Early stopping parameters
            val_loader=val_meta_loader,
            val_interval=val_interval,
            patience=patience,
            save_path=best_model_path,
        )
    elif use_v2:
        # V2: Each inner step uses a different batch
        meta_losses = meta_train_v2(
            model=model,
            meta_loader=train_meta_loader,
            optimizer=optimizer,
            inner_lr=meta_config["inner_lr"],
            inner_epochs=meta_config.get("inner_epochs", 1),
            num_outer_steps=meta_config["num_outer_steps"],
            grad_clip=meta_config["grad_clip"],
            scheduler=None,
            log_interval=meta_config["log_interval"],
            wandb_run=wandb_run,
            val_loader=val_meta_loader,
            val_interval=val_interval,
            num_val_tasks=val_tasks,
        )
    else:
        # V1: Same batch for all inner steps (with time-split)
        meta_losses = meta_train(
            model=model,
            meta_loader=train_meta_loader,
            optimizer=optimizer,
            inner_steps=meta_config["inner_steps"],
            inner_lr=meta_config["inner_lr"],
            num_outer_steps=meta_config["num_outer_steps"],
            grad_clip=meta_config["grad_clip"],
            scheduler=None,
            log_interval=meta_config["log_interval"],
            wandb_run=wandb_run,
            val_loader=val_meta_loader,
            val_interval=val_interval,
            num_val_tasks=val_tasks,
        )
    
    # Final validation
    print("\n=== Final Validation ===")
    if use_v3:
        val_loss = meta_validate_v3(
            model=model,
            meta_loader=val_meta_loader,
            inner_steps=meta_config["inner_steps"],
            inner_lr=meta_config["inner_lr"],
            num_val_tasks=20,
        )
    elif use_v2:
        val_loss = meta_validate_v2(
            model=model,
            meta_loader=val_meta_loader,
            inner_lr=meta_config["inner_lr"],
            inner_epochs=meta_config.get("inner_epochs", 1),
            num_val_tasks=20,
        )
    else:
        val_loss = meta_validate(
            model=model,
            meta_loader=val_meta_loader,
            inner_steps=meta_config["inner_steps"],
            inner_lr=meta_config["inner_lr"],
            num_val_tasks=20,
        )
    print(f"Final validation loss: {val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(foldername, "meta_model.pth"))
    
    # Save training history
    early_stopped = len(meta_losses) < meta_config["num_outer_steps"]
    history = {
        "meta_losses": meta_losses,
        "final_val_loss": val_loss,
        "early_stopped": early_stopped,
        "actual_steps": len(meta_losses),
        "max_steps": meta_config["num_outer_steps"],
    }
    with open(os.path.join(foldername, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    
    if early_stopped:
        print(f"[Early Stopping] Training stopped at step {len(meta_losses)}/{meta_config['num_outer_steps']}")
    if wandb_run is not None:
        for step, loss in enumerate(meta_losses, start=1):
            wandb_run.log({"meta/loss": loss}, step=step)
        wandb_run.log({"meta/final_val_loss": val_loss}, step=len(meta_losses))
        wandb_run.save(os.path.join(foldername, "meta_model.pth"))
        wandb_run.finish()
    
    print(f"\n=== Training Complete ===")
    print(f"Mode: {'Multi-Dataset' if multi_dataset_mode else 'Single-Dataset'}")
    print(f"Training datasets: {train_datasets}")
    if args.held_out:
        print(f"Held-out dataset: {args.held_out}")
    print(f"Task sampling: {'Round-Robin' if args.round_robin else 'Random'}")
    print(f"Meta-learning version: {version_label}")
    print(f"Meta-trained model saved to {foldername}")
    print(f"Final training loss: {meta_losses[-1]:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
