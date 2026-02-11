import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import pickle
from torch import nn
import copy
import pandas as pd


def _resolve_wandb_run(use_wandb, wandb_run):
    if not use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("W&B requested but wandb is not installed.") from exc
    run = wandb_run or wandb.run
    if run is None:
        raise ValueError("W&B requested but no active run. Call wandb.init first.")
    return run



def train_conditional_diffusion(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    use_wandb=False,
    wandb_run=None,
    gcn_lr_scale=None,
):
    wandb_run = _resolve_wandb_run(use_wandb, wandb_run)
    base_lr = config["lr"]
    if gcn_lr_scale is not None:
        gcn_lr = base_lr * gcn_lr_scale
        gcn_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "graph_layer" in name:
                gcn_params.append(param)
            else:
                other_params.append(param)
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr})
        if gcn_params:
            param_groups.append({"params": gcn_params, "lr": gcn_lr})
        optimizer = Adam(param_groups, weight_decay=1e-6)
    else:
        optimizer = Adam(model.parameters(), lr=base_lr, weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"
        best_output_path = foldername + "/best_model.pth"
    
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if wandb_run is not None:
            log_payload = {
                "train/avg_epoch_loss": avg_loss / batch_no,
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch_no,
            }
            if gcn_lr_scale is not None and len(optimizer.param_groups) > 1:
                log_payload["train/lr_gcn"] = optimizer.param_groups[1]["lr"]
            wandb_run.log(log_payload, step=epoch_no)
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            avg_loss_valid = avg_loss_valid / batch_no
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid,
                    "at",
                    epoch_no,
                )
                if foldername != "":
                    torch.save(model.state_dict(), best_output_path)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/avg_epoch_loss": avg_loss_valid,
                        "epoch": epoch_no,
                    },
                    step=epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)
        # torch.save(model_bwd.state_dict(), output_path_bwd)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def calculate_station_metrics(target, generated_samples, eval_points, scaler, mean_scaler):
    """Compute per-station RMSE, MAE, and CRPS."""
    B, L, K = target.shape
    station_metrics = []
    

    samples_median = generated_samples.median(dim=1).values  # (B, L, K)
    
    for k in range(K):

        target_k = target[:, :, k]  # (B, L)
        samples_median_k = samples_median[:, :, k]  # (B, L)
        eval_points_k = eval_points[:, :, k]  # (B, L)
        generated_samples_k = generated_samples[:, :, :, k]  # (B, nsample, L)
        

        if target_k.shape != samples_median_k.shape or target_k.shape != eval_points_k.shape:
            min_dim1 = min(target_k.shape[1], samples_median_k.shape[1], eval_points_k.shape[1])
            target_k = target_k[:, :min_dim1]
            samples_median_k = samples_median_k[:, :min_dim1]
            eval_points_k = eval_points_k[:, :min_dim1]
        

        diff = samples_median_k - target_k
        masked_diff = diff * eval_points_k
        squared_diff = masked_diff ** 2
        

        if isinstance(scaler, torch.Tensor) and scaler.shape[0] == K:
            scaler_k = scaler[k].item()
        else:
            scaler_k = scaler
            
        mse_k = squared_diff * (scaler_k ** 2)
        rmse_k = torch.sqrt(mse_k.sum() / eval_points_k.sum()) if eval_points_k.sum() > 0 else torch.tensor(0.0)
        

        mae_k = (torch.abs((samples_median_k - target_k) * eval_points_k) * scaler_k).sum() / eval_points_k.sum() if eval_points_k.sum() > 0 else torch.tensor(0.0)
        

        if isinstance(mean_scaler, torch.Tensor) and mean_scaler.shape[0] == K:
            mean_scaler_k = mean_scaler[k].item()
        else:
            mean_scaler_k = mean_scaler
        
        target_k_scaled = target_k * scaler_k + mean_scaler_k
        samples_k_scaled = generated_samples_k * scaler_k + mean_scaler_k
        
        crps_k = calc_quantile_CRPS_single_station(target_k_scaled, samples_k_scaled, eval_points_k, mean_scaler_k, scaler_k)
        
        station_metrics.append((rmse_k.item(), mae_k.item(), crps_k))
    
    return station_metrics


def calc_quantile_CRPS_single_station(target, forecast, eval_points, mean_scaler, scaler):
    """Compute CRPS for one station."""
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast, quantiles[i], dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom if denom > 0 else 0
    
    return CRPS.item() / len(quantiles) if denom > 0 else 0.0


def get_station_ids(num_stations):
    """Return station IDs for a known station count."""
    if num_stations == 20:

        return ['4178000', '4182000', '4183000', '4183500', '4184500', '4185000', '4185318', '4185440', 
                '4186500', '4188100', '4188496', '4189000', '4190000', '4191058', '4191444', '4191500', 
                '4192500', '4192574', '4192599', '4193500']
    elif num_stations == 12:

        return ['04178000', '04183000', '04183500', '04185318', '04186500', 
                '04188100', '04190000', '04191058', '04191444', '04191500', '04192500', '04193500']
    elif num_stations == 14:
        # Common stations (excluding 04182000 due to high missing rate)
        return ['04178000', '04183000', '04183500', '04184500', '04185318', '04186500', 
                '04188100', '04188496', '04190000', '04191058', '04191444', '04191500', 
                '04192500', '04193500']
    else:

        return [f"Station_{i+1}" for i in range(num_stations)]


def save_results_to_csv(all_target, all_generated_samples, all_evalpoint, all_observed_point, 
                        all_observed_time, scaler, mean_scaler, foldername, nsample):
    """Save per-station and merged prediction CSV files."""

    target_np = all_target.cpu().numpy()  # (B, L, K)
    samples_np = all_generated_samples.cpu().numpy()  # (B, nsample, L, K)
    evalpoint_np = all_evalpoint.cpu().numpy()  # (B, L, K)
    observed_point_np = all_observed_point.cpu().numpy()  # (B, L, K)
    time_np = all_observed_time.cpu().numpy()  # (B, L)
    

    predictions_np = np.median(samples_np, axis=1)  # (B, L, K)
    

    if isinstance(scaler, torch.Tensor):
        scaler = scaler.cpu().numpy()
    if isinstance(mean_scaler, torch.Tensor):
        mean_scaler = mean_scaler.cpu().numpy()
    

    target_original = target_np * scaler + mean_scaler  # (B, L, K)
    predictions_original = predictions_np * scaler + mean_scaler  # (B, L, K)
    
    B, L, K = target_np.shape
    

    station_ids = get_station_ids(K)
    

    for k in range(K):
        station_id = station_ids[k] if k < len(station_ids) else f"Station_{k+1}"
        

        data_rows = []
        
        for b in range(B):
            for l in range(L):

                time_idx = int(time_np[b, l])
                

                observed_val = target_original[b, l, k]
                

                predicted_val = predictions_original[b, l, k]
                


                mask_val = int(evalpoint_np[b, l, k])
                

                data_rows.append({
                    'timestep': b * L + l,
                    'sequence_id': b,
                    'time_in_sequence': l,
                    'imputed_value': predicted_val,
                    'ground_truth_value': observed_val,
                    'evaluation_mask': mask_val,
                    'is_observed_point': int(observed_point_np[b, l, k])
                })
        

        df = pd.DataFrame(data_rows)
        

        csv_filename = f"{foldername}/predictions_station_{station_id}_nsample{nsample}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved predictions for station {station_id} to {csv_filename}")
    

    all_data_rows = []
    for k in range(K):
        station_id = station_ids[k] if k < len(station_ids) else f"Station_{k+1}"
        for b in range(B):
            for l in range(L):
                time_idx = int(time_np[b, l])
                all_data_rows.append({
                    'station_id': station_id,
                    'timestep': b * L + l,
                    'sequence_id': b,
                    'time_in_sequence': l,
                    'imputed_value': predictions_original[b, l, k],
                    'ground_truth_value': target_original[b, l, k],
                    'evaluation_mask': int(evalpoint_np[b, l, k]),
                    'is_observed_point': int(observed_point_np[b, l, k])
                })
    
    df_all = pd.DataFrame(all_data_rows)
    csv_all_filename = f"{foldername}/predictions_all_stations_nsample{nsample}.csv"
    df_all.to_csv(csv_all_filename, index=False)
    print(f"Saved combined predictions for all stations to {csv_all_filename}")


def evaluate_conditional_diffusion(
    model,
    test_loader,
    nsample=100,
    scaler=1,
    mean_scaler=0,
    foldername="",
    device=None,
    use_wandb=False,
    wandb_run=None,
    log_station_metrics=False,
):
    """
    Evaluate conditional diffusion model.
    
    Args:
        model: conditional diffusion model
        test_loader: Test data loader
        nsample: Number of samples for imputation
        scaler: Standard deviation for denormalization
        mean_scaler: Mean for denormalization
        foldername: Folder to save results
        device: Device to use
    """
    wandb_run = _resolve_wandb_run(use_wandb, wandb_run)
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)


                if isinstance(scaler, torch.Tensor):

                    scaler_expanded = scaler.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
                    scaler_expanded = scaler_expanded.expand(c_target.shape)  # (B, L, K)
                else:
                    scaler_expanded = scaler
                
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler_expanded ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler_expanded

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )


            print("\n=== Per-station metrics ===")
            station_metrics = calculate_station_metrics(
                all_target, all_generated_samples, all_evalpoint, scaler, mean_scaler
            )
            

            station_ids = get_station_ids(all_target.shape[-1])
            

            for i, (rmse, mae, crps) in enumerate(station_metrics):
                station_id = station_ids[i] if i < len(station_ids) else f"Station_{i+1}"
                print(f"{station_id}: RMSE={rmse:.4f}, MAE={mae:.4f}, CRPS={crps:.4f}")

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("\n=== Overall metrics ===")
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)
            if wandb_run is not None:
                metrics = {
                    "eval/rmse": np.sqrt(mse_total / evalpoints_total),
                    "eval/mae": mae_total / evalpoints_total,
                    "eval/crps": CRPS,
                    "eval/crps_sum": CRPS_sum,
                    "eval/nsample": nsample,
                }
                if log_station_metrics:
                    for i, (rmse, mae, crps) in enumerate(station_metrics):
                        station_id = station_ids[i] if i < len(station_ids) else f"Station_{i+1}"
                        metrics[f"eval/station/{station_id}_rmse"] = rmse
                        metrics[f"eval/station/{station_id}_mae"] = mae
                        metrics[f"eval/station/{station_id}_crps"] = crps
                wandb_run.log(metrics)

                wandb_run.summary.update(metrics)
            

            print("\n=== Saving CSV files ===")
            save_results_to_csv(
                all_target, 
                all_generated_samples, 
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                scaler,
                mean_scaler,
                foldername,
                nsample
            )
    return {
        "rmse": np.sqrt(mse_total / evalpoints_total),
        "mae": mae_total / evalpoints_total,
        "crps": CRPS,
        "crps_sum": CRPS_sum,
        "station_metrics": station_metrics,
    }


# Backward-compatible aliases
def train_CSDI(*args, **kwargs):
    return train_conditional_diffusion(*args, **kwargs)


def evaluate_CSDI(*args, **kwargs):
    return evaluate_conditional_diffusion(*args, **kwargs)
