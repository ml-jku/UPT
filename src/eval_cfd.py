import os
import platform
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import wandb
import yaml


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--stage_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_model.loss.online.x_hat.E1")
    parser.add_argument("--mode", type=str, default="correlation", choices=["gif", "loss", "correlation"])
    parser.add_argument("--num_input_points", type=int)
    parser.add_argument("--num_input_points_ratio", type=float)
    parser.add_argument("--k", type=int)
    parser.add_argument("--rollout_mode", type=str, choices=["image", "latent"])
    parser.add_argument("--version", type=str)
    parser.add_argument("--num_supernodes", type=int)
    parser.add_argument("--resolution", type=int)
    return vars(parser.parse_args())


def main(
        stage_id,
        checkpoint,
        mode,
        num_input_points,
        num_input_points_ratio,
        k,
        rollout_mode,
        version,
        num_supernodes,
        resolution,
):
    # init args + W&B
    print(f"stage_id: {stage_id}")
    print(f"checkpoint: {checkpoint}")
    print(f"mode: {mode} (gif or loss)")
    print(f"num_input_points: {num_input_points}")
    print(f"num_input_points_ratio: {num_input_points_ratio}")
    print(f"num_supernodes: {num_supernodes}")
    print(f"resolution: {resolution}")
    wandb.login(host="https://api.wandb.ai/")
    api = wandb.Api()

    # generate yamls
    out = Path("yamls_run")
    out.mkdir(exist_ok=True)
    print(stage_id)
    run = api.run(f"jku-ssl/cvsim/{stage_id}")
    name = "/".join(run.name.split("/")[:-1])
    if num_input_points_ratio is not None:
        name += f"-subsam{str(num_input_points_ratio).replace('.', '')}"
    if k is not None:
        name += f"-k{k}"

    if "grid_resolution" in run.config["datasets"]["train"]:
        if "standardize_query_pos" in run.config["datasets"]["train"]:
            print(f"found grid_resolution and standardize_query_pos -> using interpolated template")
            template_fname = "interpolated"
        else:
            print("found grid_resolution -> using gino template")
            template_fname = "gino"
    else:
        print("found no grid_resolution -> using simformer template")
        template_fname = "simformer"

    template_uri = f"yamls/eval/cfd/{mode}/{template_fname}.yaml"
    with open(template_uri) as f:
        hp = yaml.safe_load(f)

    # fetch mode
    if rollout_mode is None:
        rec_prev_x = run.config["trainer"].get("reconstruct_prev_x_weight", 0)
        rec_dynamics = run.config["trainer"].get("reconstruct_dynamics_weight", 0)
        if rec_prev_x > 0 or rec_dynamics > 0:
            print(f"found reconstruction loss -> use mode=latent")
            rollout_mode = "latent"
        else:
            print(f"no reconstruction losses found -> use mode=image")
            rollout_mode = "image"
    hp["vars"]["mode"] = rollout_mode
    name += f"-{rollout_mode}"
    name += f"-{checkpoint.split('_')[0]}"
    if num_input_points is not None:
        if num_input_points > 1000:
            name += f"_in{num_input_points // 1000}k"
        else:
            name += f"_in{num_input_points}"
    if num_supernodes is not None:
        name += f"_{num_supernodes}supernodes"
    if resolution is not None:
        name += f"_{resolution}resolution"

    # set other params
    hp["vars"]["stage_id"] = stage_id
    if checkpoint.isdigit():
        hp["vars"]["checkpoint"] = dict(epoch=checkpoint)
    elif checkpoint.startswith("E") and checkpoint[1:].isdigit():
        hp["vars"]["checkpoint"] = dict(epoch=checkpoint[1:])
    else:
        hp["vars"]["checkpoint"] = checkpoint
    if version is None:
        hp["vars"]["version"] = run.config["datasets"]["train"]["version"]
    else:
        hp["vars"]["version"] = version
        name += f"-{name}"
    hp["vars"]["num_input_timesteps"] = run.config["datasets"]["train"]["num_input_timesteps"]
    if num_input_points_ratio is None and "num_input_points_ratio" not in run.config["datasets"]["train"]:
        hp["vars"]["num_input_points_ratio"] = None
    else:
        if num_input_points_ratio is not None:
            hp["vars"]["num_input_points_ratio"] = num_input_points_ratio
        else:
            hp["vars"]["num_input_points_ratio"] = run.config["datasets"]["train"]["num_input_points_ratio"]
    if num_input_points is None and "num_input_points" not in run.config["datasets"]["train"]:
        hp["vars"]["num_input_points"] = None
    else:
        if num_input_points is not None:
            hp["vars"]["num_input_points"] = num_input_points
        else:
            # wandb stores lists as dictionary with indices as keys
            # if num_input_points is sampled -> force --num_input_points
            if isinstance(run.config["datasets"]["train"]["num_input_points"], dict):
                assert "test_rollout" in run.config["datasets"]
                hp["vars"]["num_input_points"] = run.config["datasets"]["test_rollout"]["num_input_points"]
            else:
                hp["vars"]["num_input_points"] = run.config["datasets"]["train"]["num_input_points"]
    if "radius_graph_r" in run.config["datasets"]["train"]:
        hp["vars"]["radius_graph_r"] = run.config["datasets"]["train"]["radius_graph_r"]
        radius_graph_max_num_neighbors = k or run.config["datasets"]["train"]["radius_graph_max_num_neighbors"]
        hp["vars"]["radius_graph_max_num_neighbors"] = radius_graph_max_num_neighbors
    else:
        hp["vars"]["radius_graph_r"] = run.config["trainer"]["radius_graph_r"]
        hp["vars"]["radius_graph_max_num_neighbors"] = k or run.config["trainer"]["radius_graph_max_num_neighbors"]
    if "norm" in run.config["datasets"]["train"]:
        hp["vars"]["norm"] = run.config["datasets"]["train"]["norm"]
    else:
        hp["vars"]["norm"] = "mean0std1"
    if "clamp" in run.config["datasets"]["train"]:
        hp["vars"]["clamp"] = run.config["datasets"]["train"]["clamp"]
    else:
        hp["vars"]["clamp"] = None
    if "clamp_mode" in run.config["datasets"]["train"]:
        hp["vars"]["clamp_mode"] = run.config["datasets"]["train"]["clamp_mode"]
    else:
        hp["vars"]["clamp_mode"] = "hard"
    if "max_num_timesteps" in run.config["datasets"]["train"]:
        hp["vars"]["max_num_timesteps"] = run.config["datasets"]["train"]["max_num_timesteps"]
    else:
        hp["vars"]["max_num_timesteps"] = None
    hp["trainer"]["precision"] = run.config["trainer"]["precision"]
    if "backup_precision" in run.config["trainer"]:
        hp["trainer"]["backup_precision"] = run.config["trainer"]["backup_precision"]
    if num_supernodes is None:
        if "num_supernodes" in run.config["datasets"]["train"]:
            hp["vars"]["num_supernodes"] = run.config["datasets"]["train"]["num_supernodes"]
        elif "num_supernodes" in run.config["datasets"]["train"]["collators"]["0"]:
            hp["vars"]["num_supernodes"] = run.config["datasets"]["train"]["collators"]["0"]["num_supernodes"]
        else:
            if "num_supernodes" in hp["vars"]:
                hp["vars"]["num_supernodes"] = None
    else:
        hp["vars"]["num_supernodes"] = num_supernodes
    if resolution is None:
        if "grid_resolution" in run.config["datasets"]["train"]:
            # wandb stores lists as dictionary with indices as keys
            resolution = [
                run.config["datasets"]["train"]["grid_resolution"]["0"],
                run.config["datasets"]["train"]["grid_resolution"]["1"],
            ]
            hp["vars"]["grid_resolution"] = resolution
    else:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]
        hp["vars"]["grid_resolution"] = resolution
    hp["name"] = hp["name"].replace("???", name)
    fname = f"{stage_id}_rollout_{template_fname}_{mode}_{rollout_mode}_{checkpoint.split('_')[0]}"
    if num_input_points_ratio is not None:
        fname += f"_subsam{str(num_input_points_ratio).replace('.', '')}"
    if num_input_points is not None:
        if num_input_points > 1000:
            fname += f"_in{num_input_points // 1000}k"
        else:
            fname += f"_in{num_input_points}"
    if num_supernodes is not None:
        fname += f"_{num_supernodes}supernodes"
    if resolution is not None:
        if isinstance(resolution, int):
            fname += f"_{resolution}resolution"
        else:
            fname += f"_{resolution[0]}resolution"
    if k is not None:
        fname += f"_k{k}"
    out_uri = out / f"{fname}.yaml"
    with open(out_uri, "w") as f:
        yaml.safe_dump(hp, f, sort_keys=False)
    print(f"created '{out_uri.as_posix()}'")
    wandb.finish()


if __name__ == "__main__":
    main(**parse_args())
