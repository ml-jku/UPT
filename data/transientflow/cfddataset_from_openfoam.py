import os
import pickle
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="/OpenFOAM/")
    parser.add_argument("--dst", type=str, required=True, help="/publicdata/")
    parser.add_argument("--num_workers", type=int, required=True)
    return vars(parser.parse_args())


class CopyDataset(Dataset):
    def __init__(self, src, dst, case_names):
        super().__init__()
        self.src = src
        self.dst = dst
        self.case_names = case_names

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        src = self.src / case_name
        dst = self.dst / case_name
        if dst.exists():
            return 1
        else:
            dst.mkdir()

        fnames = [(True, "x.th"), (True, "y.th"), (False, "object_mask.th")]
        fnames += [(True, f"{i:08d}_mesh.th") for i in range(120)]
        for to_fp16, fname in fnames:
            if not (src / fname).exists():
                print(f"file not found: {(src / fname).as_posix()}")
                return 1
            if to_fp16:
                data = torch.load(src / fname).half()
                torch.save(data, dst / fname)
            else:
                shutil.copyfile(src / fname, dst / fname)
        with open(src / f"simulation_description.pkl", "rb") as f:
            desc = pickle.load(f)
            uinit = desc["initial_velocity"]
            num_objects = desc["n_objects"]
        uinit_uri = dst / f"U_init.th"
        torch.save(torch.tensor(uinit), uinit_uri)
        num_objects_uri = dst / f"num_objects.th"
        torch.save(torch.tensor(num_objects), num_objects_uri)
        return 0


def main(src, dst, num_workers):
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()

    dst.mkdir(exist_ok=True)

    print(f"src: {src.as_posix()}")
    print(f"dst: {dst.as_posix()}")

    case_names = list(sorted([case_name for case_name in os.listdir(src) if "case_" in case_name]))
    print(f"found {len(case_names)} case_names")
    dataset = CopyDataset(src=src, dst=dst, case_names=case_names)
    for _ in tqdm(DataLoader(dataset, batch_size=1, num_workers=num_workers)):
        pass
    print("fin")


if __name__ == "__main__":
    main(**parse_args())
