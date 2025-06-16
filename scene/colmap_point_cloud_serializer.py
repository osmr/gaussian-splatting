import numpy as np
from scene.colmap_point_cloud import ColmapPointCloud
from scene.colmap_utils import colmap_binary_read_next_bytes


class ColmapPointCloudSerializer:

    @staticmethod
    def load_from_txt(txt_file_path: str) -> ColmapPointCloud:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DText(const std::string& path)
            void Reconstruction::WritePoints3DText(const std::string& path)
        """
        num_points = 0
        with open(txt_file_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    num_points += 1

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        count = 0
        with open(txt_file_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elements = line.split()
                    xyz = np.array(tuple(map(float, elements[1:4])))
                    rgb = np.array(tuple(map(int, elements[4:7])))
                    error = np.array(float(elements[7]))
                    xyzs[count] = xyz
                    rgbs[count] = rgb
                    errors[count] = error
                    count += 1

        return ColmapPointCloud(
            points=xyzs,
            colors=rgbs,
            errors=errors)

    @staticmethod
    def load_from_bin(bin_file_path: str) -> ColmapPointCloud:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)
        """
        with open(bin_file_path, "rb") as fid:
            num_points = colmap_binary_read_next_bytes(
                fid=fid,
                num_bytes=8,
                format_char_sequence="Q")[0]
            xyzs = np.empty((num_points, 3))
            rgbs = np.empty((num_points, 3))
            errors = np.empty((num_points, 1))

            for p_id in range(num_points):
                binary_point_line_properties = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=43,
                    format_char_sequence="QdddBBBd")
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=8,
                    format_char_sequence="Q")[0]
                _ = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=(8 * track_length),
                    format_char_sequence=("ii" * track_length))
                xyzs[p_id] = xyz
                rgbs[p_id] = rgb
                errors[p_id] = error

        return ColmapPointCloud(
            points=xyzs,
            colors=rgbs,
            errors=errors)
