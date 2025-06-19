import numpy as np
from scene.colmap_point_cloud import ColmapPointCloud
from scene.colmap_utils import colmap_binary_read_next_bytes


class ColmapPointCloudSerializer:

    @staticmethod
    def load_from_txt(txt_file_path: str) -> ColmapPointCloud:
        """
        Load Colmap point cloud from text file.
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DText(const std::string& path)
            void Reconstruction::WritePoints3DText(const std::string& path)

        Parameters
        ----------
        txt_file_path : str
            Source text file path.

        Returns
        -------
        ColmapPointCloud
            Resulted Colmap point cloud.
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

        pts_positions = np.empty((num_points, 3))
        pts_colors = np.empty((num_points, 3))
        pts_errors = np.empty((num_points, 1))

        pt_idx = 0
        with open(txt_file_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if (len(line) > 0) and (line[0] != "#"):
                    pt_info = line.split()
                    pt_position = np.array(tuple(map(float, pt_info[1:4])))
                    pt_color = np.array(tuple(map(int, pt_info[4:7])))
                    pt_error = np.array(float(pt_info[7]))
                    pts_positions[pt_idx] = pt_position
                    pts_colors[pt_idx] = pt_color
                    pts_errors[pt_idx] = pt_error
                    pt_idx += 1

        return ColmapPointCloud(
            positions=pts_positions,
            colors=pts_colors,
            errors=pts_errors)

    @staticmethod
    def load_from_bin(bin_file_path: str) -> ColmapPointCloud:
        """
        Load Colmap point cloud from binary file.
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)

        Parameters
        ----------
        bin_file_path : str
            Source binary file path.

        Returns
        -------
        ColmapPointCloud
            Resulted Colmap point cloud.
        """
        with open(bin_file_path, "rb") as fid:
            num_points = colmap_binary_read_next_bytes(
                fid=fid,
                num_bytes=8,
                format_char_sequence="Q")[0]

            pts_positions = np.empty((num_points, 3))
            pts_colors = np.empty((num_points, 3))
            pts_errors = np.empty((num_points, 1))

            for pt_idx in range(num_points):
                pt_info = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=43,
                    format_char_sequence="QdddBBBd")
                pt_position = np.array(pt_info[1:4])
                pt_color = np.array(pt_info[4:7])
                pt_error = np.array(pt_info[7])
                track_length = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=8,
                    format_char_sequence="Q")[0]
                _ = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=(8 * track_length),
                    format_char_sequence=("ii" * track_length))
                pts_positions[pt_idx] = pt_position
                pts_colors[pt_idx] = pt_color
                pts_errors[pt_idx] = pt_error

        return ColmapPointCloud(
            positions=pts_positions,
            colors=pts_colors,
            errors=pts_errors)
