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
    parser.add_argument("--q", type=float, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--exclude_last", type=int, default=0)
    return vars(parser.parse_args())


def get_torch_files(root):
    result = []
    for fname in os.listdir(root):
        uri = root / fname
        if uri.is_dir():
            result += get_torch_files(uri)
        else:
            if (
                    uri.name.endswith(".th")
                    and not uri.name.startswith("coordinates")
                    and not uri.name.startswith("geometry2d")
                    and not uri.name.startswith("object_mask")
                    and not uri.name.startswith("U_init")
                    and not uri.name.startswith("num_objects")
                    and not uri.name.startswith("x")
                    and not uri.name.startswith("y")
                    and not uri.name.startswith("edge_index")
                    and not uri.name.startswith("movement_per_position")
                    and not uri.name.startswith("sampling_weights")
            ):
                try:
                    _ = int(uri.name[:len("00000000")])
                except:
                    print(f"{uri.name} is not a data file")
                    raise
                result.append(uri)
    return result


class MeanVarDataset(Dataset):
    def __init__(self, case_uris, q):
        super().__init__()
        self.case_uris = case_uris
        self.q = q

    def __len__(self):
        return len(self.case_uris)

    def __getitem__(self, idx):
        case_uri = self.case_uris[idx]
        assert case_uri.name.startswith("case_")
        uris = get_torch_files(case_uri)
        if len(uris) != 120:
            #print(f"invalid number of uris for case '{case_uri.as_posix()}' len={len(uris)}")
            raise RuntimeError(f"invalid number of uris for case '{case_uri.as_posix()}' len={len(uris)}")
        data = torch.stack([torch.load(uri) for uri in uris])
        mean = torch.zeros(3)
        var = torch.zeros(3)
        mmin = torch.zeros(3)
        mmax = torch.zeros(3)
        within1std = torch.zeros(3)
        within2std = torch.zeros(3)
        within3std = torch.zeros(3)
        for i in range(3):
            cur_data = data[:, i]
            if self.q > 0:
                # quantile is not supported for large dimensions
                # qmin = torch.quantile(cur_data, q=self.q)
                # qmax = torch.quantile(cur_data, q=1 - self.q)
                # approximate quantile by assuming a normal distribution
                cur_mean = cur_data.mean()
                cur_std = cur_data.std()
                dist = torch.distributions.Normal(loc=0, scale=1)
                qmin = cur_mean + cur_std * dist.icdf(torch.tensor(self.q))
                qmax = cur_mean + cur_std * dist.icdf(torch.tensor(1 - self.q))
                is_valid = torch.logical_and(qmin < cur_data, cur_data < qmax)
                valid_data = cur_data[is_valid]
            else:
                valid_data = cur_data
            mean[i] = valid_data.mean()
            var[i] = valid_data.var()
            mmin[i] = valid_data.min()
            mmax[i] = valid_data.max()
            cur_std = valid_data.std()
            is_within1std = torch.logical_and(mean[i] - 1 * cur_std < valid_data, valid_data < mean[i] + 1 * cur_std)
            within1std[i] = is_within1std.sum() / is_within1std.numel()
            is_within2std = torch.logical_and(mean[i] - 2 * cur_std < valid_data, valid_data < mean[i] + 2 * cur_std)
            within2std[i] = is_within2std.sum() / is_within2std.numel()
            is_within3std = torch.logical_and(mean[i] - 3 * cur_std < valid_data, valid_data < mean[i] + 3 * cur_std)
            within3std[i] = is_within3std.sum() / is_within3std.numel()
        # old impl
        # mean = torch.mean(data, dim=[0, 2])
        # var = torch.var(data, dim=[0, 2])
        # mmin = data.min(dim=2).values.min(dim=0).values
        # mmax = data.max(dim=2).values.max(dim=0).values
        return mean, var, mmin, mmax, within1std, within2std, within3std


def main(root, num_workers, exclude_last, q):
    root = Path(root).expanduser()
    assert root.exists() and root.is_dir()
    print(f"root: {root.as_posix()}")
    print(f"num_workers: {num_workers}")
    print(f"exclude_last: {exclude_last}")
    assert q < 0.5
    print(f"q (exclude values below/above quantile): {q}")

    # get all case uris
    case_uris = [root / fname for fname in os.listdir(root)]
    # sort by case index
    case_uris = list(sorted(case_uris, key=lambda cu: int(cu.name.replace("case_", ""))))
    # exclude last
    if exclude_last > 0:
        case_uris = case_uris[:-exclude_last]
    print(f"using {len(case_uris)} uris")
    print(f"last used case_uri: {case_uris[-1].as_posix()}")

    # setup dataset
    dataset = MeanVarDataset(case_uris=case_uris, q=q)

    # calculate mean/var per simulation and then average over them
    sum_of_means = 0.
    sum_of_vars = 0.
    min_of_mins = torch.full(size=(3,), fill_value=torch.inf)
    max_of_maxs = torch.full(size=(3,), fill_value=-torch.inf)
    within1std_sum = torch.zeros(3)
    within2std_sum = torch.zeros(3)
    within3std_sum = torch.zeros(3)
    for data in tqdm(DataLoader(dataset, batch_size=1, num_workers=num_workers)):
        mean, var, mmin, mmax, within1std, within2std, within3std = data
        sum_of_means += mean.squeeze(0)
        sum_of_vars += var.squeeze(0)
        min_of_mins = torch.minimum(min_of_mins, mmin.squeeze(0))
        max_of_maxs = torch.maximum(max_of_maxs, mmax.squeeze(0))
        within1std_sum += within1std.squeeze(0)
        within2std_sum += within2std.squeeze(0)
        within3std_sum += within3std.squeeze(0)

    # average
    mean = sum_of_means / len(dataset)
    std = torch.sqrt(sum_of_vars / len(dataset))
    within1std_mean = within1std_sum / len(dataset)
    within2std_mean = within2std_sum / len(dataset)
    within3std_mean = within3std_sum / len(dataset)

    #
    print(f"data_mean: {mean.tolist()}")
    print(f"data_std: {std.tolist()}")
    print(f"data_min: {min_of_mins.tolist()}")
    print(f"data_max: {max_of_maxs.tolist()}")
    print(f"within1std: {within1std_mean.tolist()}")
    print(f"within2std: {within2std_mean.tolist()}")
    print(f"within3std: {within3std_mean.tolist()}")


if __name__ == "__main__":
    main(**parse_args())
