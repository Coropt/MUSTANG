import argparse
import torch
print(torch.__version__)
cuda_available = torch.cuda.is_available()
print(cuda_available)
if cuda_available:
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    print(torch.version.cuda)
else:
    print("CUDA not available, using CPU.")
import datetime
import json
import yaml
import os
import time
import pandas as pd


from dataset_process.dataset_discharge import get_dataloader_discharge
from dataset_process.dataset_pooled import get_dataloader_pooled
from dataset_process.dataset_no3 import get_dataloader_NO3
from dataset_process.dataset_nh4 import get_dataloader_NH4
from dataset_process.dataset_srp import get_dataloader_SRP
from dataset_process.dataset_tp import get_dataloader_TP
from dataset_process.dataset_ssc import get_dataloader_SSC
from models.conditional_diffusion import ConditionalDiffusionImputation
from models.graph_aware_conditional_diffusion import GraphAwareConditionalDiffusionImputation
from util.utils import train_conditional_diffusion, evaluate_conditional_diffusion


def _str2bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "t"}

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


parser = argparse.ArgumentParser(description="conditional diffusion")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--training_missing", type=str, default="block", choices=["point", "block"]
)
parser.add_argument("--test_missing", type=str, default="block", choices=["point", "block"])  # block|point
parser.add_argument(
    "--dataset", type=str, default="air_quality", choices=["discharge", "pooled", "nh4", "no3", "srp", "tp", "ssc"]
)

parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--pretrain_missingrate",
    type=float,
    default=None,
    help="Target overall missing rate applied before building observed_mask.",
)
parser.add_argument("--sequence_length", type=int, default=16)
parser.add_argument("--use_wandb", type=_str2bool, default=False)
parser.add_argument("--wandb_project", type=str, default="mustang-conditional-diffusion")
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--pretrained_model", type=str, default=None, 
                    help="Path to pretrained model for fine-tuning (e.g., from MUSTANG)")
parser.add_argument("--epochs", type=int, default=None,
                    help="Override number of training epochs from config")
parser.add_argument("--use_graph", action="store_true",
                    help="Use graph-aware conditional diffusion spatial layer.")
parser.add_argument("--use_gcn", dest="use_graph", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--adj_path", type=str, default="",
                    help="Path to adjacency/flow-direction CSV for GCN.")
parser.add_argument("--freeze_temporal_only", action="store_true",
                    help="Freeze all params except temporal transformer (time_layer).")
parser.add_argument("--gcn_lr_scale", type=float, default=1.0,
                    help="Scale factor for GCN learning rate during training.")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

# config["model"]["is_unconditional"] = args.unconditional
config["model"]["training_missing"] = args.training_missing
config["model"]["test_missing"] = args.test_missing

config["train"]["dataset"] = args.dataset
config["train"]["seed"] = args.seed
if args.epochs is not None:
    config["train"]["epochs"] = args.epochs

# config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))
if args.pretrain_missingrate is not None and args.pretrain_missingrate > 0:
    print(
        f"[INFO] pretrain_missingrate={args.pretrain_missingrate} is set: "
        "eval_mask/testmissingratio is ignored; evaluation uses pretrained-masked points."
    )
use_graph = args.use_graph
if use_graph:
    config["diffusion"]["graph_model"] = "graph_aware"

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    "./save/" + str(args.dataset) + "/" + "/conditional_diffusion" + "-" + "training_missing" + "_"+  str(args.training_missing) + "test_missing" + "_"+  str(args.test_missing) + "_" + current_time + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

wandb_run = None
is_finetune = args.pretrained_model is not None
if args.use_wandb:
    import wandb

    stage = "finetune" if is_finetune else "baseline"
    run_name = args.wandb_name or f"{stage}_{args.dataset}_{current_time}"
    wandb_kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "config": {
            "stage": stage,
            "dataset": args.dataset,
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "training_missing": args.training_missing,
            "test_missing": args.test_missing,
            "testmissingratio": args.testmissingratio,
            "pretrain_missingrate": args.pretrain_missingrate,
            "nsample": args.nsample,
            "batch_size": config["train"]["batch_size"],
            "lr": config["train"]["lr"],
            "epochs": config["train"]["epochs"],
            "pretrained_model": args.pretrained_model,
            "model_variant": "graph-aware conditional diffusion" if use_graph else "conditional diffusion",
            "adj_path": args.adj_path,
            "freeze_temporal_only": args.freeze_temporal_only,
            "gcn_lr_scale": args.gcn_lr_scale,
        },
        "tags": [stage, "conditional-diffusion", args.dataset],
    }
    if args.wandb_group:
        wandb_kwargs["group"] = args.wandb_group
    wandb_run = wandb.init(**wandb_kwargs)

if args.dataset == "discharge":
    get_dataloader = get_dataloader_discharge
elif args.dataset == "pooled":
    get_dataloader = get_dataloader_pooled
elif args.dataset == "nh4":
    get_dataloader = get_dataloader_NH4
elif args.dataset == "no3":
    get_dataloader = get_dataloader_NO3
elif args.dataset == "srp":
    get_dataloader = get_dataloader_SRP
elif args.dataset == "tp":
    get_dataloader = get_dataloader_TP   
elif args.dataset == "ssc":
    get_dataloader = get_dataloader_SSC


train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    sequence_length=args.sequence_length,
    device=args.device,
    training_missing=args.training_missing,
    test_missing=args.test_missing,
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=args.testmissingratio,
    pretrain_missing_rate=args.pretrain_missingrate,
)

# All datasets now use the common station set defined above.
target_dim = len(COMMON_STATIONS_12)

adj_path = None
if use_graph:
    adj_path = args.adj_path or os.path.join(
        "original_data", "Discharge", "SSC_sites_flow_direction.csv"
    )
    adj = _load_flow_direction(adj_path, COMMON_STATIONS_12)
    print(f"Using graph-aware conditional diffusion adjacency from: {adj_path}")

if use_graph:
    model = GraphAwareConditionalDiffusionImputation(
        config, args.device, target_dim=target_dim, adj=adj
    ).to(args.device)
else:
    model = ConditionalDiffusionImputation(config, args.device, target_dim=target_dim).to(args.device)

# Load pretrained model if specified (for fine-tuning)
if args.pretrained_model is not None:
    print(f"\n{'='*60}")
    print(f"Loading pretrained model from: {args.pretrained_model}")
    print(f"{'='*60}")
    
    if not os.path.exists(args.pretrained_model):
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")
    
    state_dict = torch.load(args.pretrained_model, map_location=args.device)
    model.load_state_dict(state_dict)
    print(f"Successfully loaded pretrained model!")
    print(f"Fine-tuning mode: training on {args.dataset} with pretrained initialization")
    print(f"{'='*60}\n")

if args.freeze_temporal_only:
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = "time_layer" in name
        if param.requires_grad:
            trainable.append(name)
    print(f"Trainable params (temporal only): {len(trainable)}")
    for name in trainable:
        print(f"  - {name}")

gcn_lr_scale = None
if use_graph and args.gcn_lr_scale != 1.0:
    gcn_lr_scale = args.gcn_lr_scale

start_time = time.time()
print("Start time:", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

train_conditional_diffusion(
    model,
    config["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=foldername,
    use_wandb=args.use_wandb,
    wandb_run=wandb_run,
    gcn_lr_scale=gcn_lr_scale,
    )
# else:
#     model.load_state_dict(torch.load("./save" + str(args.dataset) + "/" + str(args.datatype) + "/" + args.modelfolder + "/model.pth"))

evaluate_conditional_diffusion(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    use_wandb=args.use_wandb,
    wandb_run=wandb_run,
)
end_time = time.time()
print("End time:", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
total_seconds = end_time - start_time
hours = total_seconds // 3600
remaining_seconds = total_seconds % 3600
minutes = remaining_seconds // 60
seconds = remaining_seconds % 60
print(
    "Total time: {:.0f}h {:.0f}min {:.4f}s".format(
        hours, minutes, seconds
    )
)
if wandb_run is not None:
    wandb_run.save(os.path.join(foldername, "model.pth"))
    wandb_run.finish()
