import os
from argparse import ArgumentParser
from pathlib import Path

import einops
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="e.g. /system/user/publicdata/CVSim/mesh_dataset/v1")
    parser.add_argument("--num_workers", type=int, default=0)
    return vars(parser.parse_args())


class NumNodesDataset(Dataset):
    def __init__(self, case_uris):
        super().__init__()
        self.case_uris = case_uris

    def __len__(self):
        return len(self.case_uris)

    def __getitem__(self, idx):
        case_uri = self.case_uris[idx]
        assert case_uri.name.startswith("case_")
        return len(torch.load(case_uri / "x.th"))

def main(root, num_workers):
    root = Path(root).expanduser()
    assert root.exists() and root.is_dir()
    print(f"root: {root.as_posix()}")
    print(f"num_workers: {num_workers}")

    # get all case uris
    case_uris = [root / fname for fname in os.listdir(root)]
    # sort by case index
    case_uris = list(sorted(case_uris, key=lambda cu: int(cu.name.replace("case_", ""))))
    print(f"using {len(case_uris)} uris")
    print(f"last used case_uri: {case_uris[-1].as_posix()}")

    # setup dataset
    dataset = NumNodesDataset(case_uris=case_uris)

    # calculate mean/var per simulation and then average over them
    num_nodes = []
    for data in tqdm(DataLoader(dataset, batch_size=1, num_workers=num_workers)):
        num_nodes.append(data.clone())
    num_nodes = torch.concat(num_nodes)
    #
    print(f"mean: {num_nodes.float().mean()}")
    print(f"min: {num_nodes.min()}")
    print(f"max: {num_nodes.max()}")
    print(f"<32K: {(num_nodes < 32768).sum()}")


if __name__ == "__main__":
    main(**parse_args())
