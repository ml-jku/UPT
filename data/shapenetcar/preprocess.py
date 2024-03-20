# conda create --name open3d python=3.9
# pip install open3d
# pip install meshio
# pip install torch
# pip install tempfile
import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import meshio
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="e.g. /data/shapenet_car/training_data")
    parser.add_argument("--dst", type=str, required=True, help="e.g. /data/shapenet_car/preprocessed")
    return vars(parser.parse_args())


def sdf(mesh, resolution):
    quads = mesh.cells_dict["quad"]

    idx = np.flatnonzero(quads[:, -1] == 0)
    out0 = np.empty((quads.shape[0], 2, 3), dtype=quads.dtype)

    out0[:, 0, 1:] = quads[:, 1:-1]
    out0[:, 1, 1:] = quads[:, 2:]

    out0[..., 0] = quads[:, 0, None]

    out0.shape = (-1, 3)

    mask = np.ones(out0.shape[0], dtype=bool)
    mask[idx * 2 + 1] = 0
    quad_to_tri = out0[mask]

    cells = [("triangle", quad_to_tri)]

    new_mesh = meshio.Mesh(mesh.points, cells)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as tf:
        new_mesh.write(tf, file_format="ply")
        open3d_mesh = o3d.io.read_triangle_mesh(tf.name)
    open3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(open3d_mesh)

    domain_min = torch.tensor([-2.0, -1.0, -4.5])
    domain_max = torch.tensor([2.0, 4.5, 6.0])
    tx = np.linspace(domain_min[0], domain_max[0], resolution)
    ty = np.linspace(domain_min[1], domain_max[1], resolution)
    tz = np.linspace(domain_min[2], domain_max[2], resolution)
    grid = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)
    return torch.from_numpy(scene.compute_signed_distance(grid).numpy()).float()


def main(src, dst):
    src = Path(src).expanduser()
    assert src.exists(), f"'{src.as_posix()}' doesnt exist"
    assert src.name == "training_data"
    dst = Path(dst).expanduser()
    # assert not dst.exists(), f"'{dst.as_posix()}' exist"
    print(f"src: {src.as_posix()}")
    print(f"dst: {dst.as_posix()}")

    # collect uris for samples
    uris = []
    for i in range(9):
        param_uri = src / f"param{i}"
        for name in sorted(os.listdir(param_uri)):
            # param folders contain .npy/.py/txt files
            if "." in name:
                continue
            potential_uri = param_uri / name
            assert potential_uri.is_dir()
            uris.append(potential_uri)
    print(f"found {len(uris)} samples")

    # .vtk files contains points that dont belong to the mesh -> filter them out
    mesh_point_counts = []
    for uri in tqdm(uris):
        reluri = uri.relative_to(src)
        out = dst / reluri
        out.mkdir(exist_ok=True, parents=True)

        # filter out mesh points that are not part of the shape
        mesh = meshio.read(uri / "quadpress_smpl.vtk")
        assert len(mesh.cells) == 1
        cell_block = mesh.cells[0]
        assert cell_block.type == "quad"
        unique = np.unique(cell_block.data)
        mesh_point_counts.append(len(unique))
        mesh_points = torch.from_numpy(mesh.points[unique]).float()
        pressure = torch.from_numpy(np.load(uri / "press.npy")[unique]).float()
        torch.save(mesh_points, out / "mesh_points.th")
        torch.save(pressure, out / "pressure.th")

        # generate sdf
        for resolution in [32, 40, 48, 64, 80]:
            torch.save(sdf(mesh, resolution=resolution), out / f"sdf_res{resolution}.th")

    print("fin")


if __name__ == "__main__":
    main(**parse_args())
