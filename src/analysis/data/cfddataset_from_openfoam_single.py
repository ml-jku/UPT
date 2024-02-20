import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
import pickle

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="/OpenFOAM/case_XX")
    parser.add_argument("--dst", type=str, required=True, help="/publicdata/case_XX")
    return vars(parser.parse_args())


def main(src, dst):
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()

    print(f"src: {src.as_posix()}")
    print(f"dst: {dst.as_posix()}")

    assert "case" in src.name
    assert "case" in dst.name
    assert src.exists()

    if not dst.exists():
        dst.mkdir()
    else:
        print("dst already exists -> return")
        return
    fnames = [(True, "x.th"), (True, "y.th"), (False, "object_mask.th")]
    fnames += [(True, f"{i:08d}_mesh.th") for i in range(120)]
    print(f"copy {len(fnames) - 3} trajectories, x, y, object_mask")
    for to_fp16, fname in tqdm(fnames):
        if to_fp16:
            data = torch.load(src / fname).half()
            torch.save(data, dst / fname)
        else:
            shutil.copyfile(src / fname, dst / fname)
    with open(src / f"simulation_description.pkl", "rb") as f:
        uinit = pickle.load(f)["initial_velocity"]
    uinit_uri = dst / f"U_init.th"
    print(f"create {uinit_uri.as_posix()}")
    torch.save(torch.tensor(uinit), uinit_uri)
    print("fin")


if __name__ == "__main__":
    main(**parse_args())
