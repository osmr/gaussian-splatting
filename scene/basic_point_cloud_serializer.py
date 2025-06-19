import numpy as np
from plyfile import PlyData, PlyElement
from scene.basic_point_cloud import BasicPointCloud


class BasicPointCloudSerializer:

    @staticmethod
    def load_from_ply(ply_file_path: str) -> BasicPointCloud:
        ply_data = PlyData.read(ply_file_path)
        vertices = ply_data["vertex"]

        positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T

        return BasicPointCloud(
            positions=positions,
            colors=colors,
            normals=normals)

    @staticmethod
    def save_to_ply(data: BasicPointCloud,
                    ply_file_path: str):
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")
        ]

        normals = data.normals if data.normals else np.zeros_like(data.positions)
        colors = np.clip(data.colors * 255.0, a_min=0.0, a_max=255.0).astype(np.uint8)

        elements = np.empty(data.positions.shape[0], dtype=dtype)
        attributes = np.hstack((data.positions, normals, colors))
        elements[:] = list(map(tuple, attributes))

        vertex_element = PlyElement.describe(
            data=elements,
            name="vertex")

        ply_data = PlyData([vertex_element])
        ply_data.write(ply_file_path)
