import numpy as np
from scene.colmap_camera_extrinsic import ColmapCameraExtrinsic
from scene.colmap_utils import colmap_binary_read_next_bytes


class ColmapCameraExtrinsicSerializer:

    @staticmethod
    def load_from_txt(txt_file_path: str) -> dict[int, ColmapCameraExtrinsic]:
        """
        Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
        """
        camera_extrinsics = {}
        with open(txt_file_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if (len(line) > 0) and (line[0] != "#"):
                    image_properties = line.split()
                    image_id = int(image_properties[0])
                    qvec = np.array(tuple(map(float, image_properties[1:5])))
                    tvec = np.array(tuple(map(float, image_properties[5:8])))
                    camera_id = int(image_properties[8])
                    image_file_name = image_properties[9]
                    pts_infos = fid.readline().split()
                    pts2d = np.column_stack([tuple(map(float, pts_infos[0::3])), tuple(map(float, pts_infos[1::3]))])
                    pts3d_ids = np.array(tuple(map(int, pts_infos[2::3])))
                    camera_extrinsics[image_id] = ColmapCameraExtrinsic(
                        image_id=image_id,
                        qvec=qvec,
                        tvec=tvec,
                        camera_id=camera_id,
                        image_file_name=image_file_name,
                        pts2d=pts2d,
                        pts3d_ids=pts3d_ids)
        return camera_extrinsics

    @staticmethod
    def load_from_bin(bin_file_path: str) -> dict[int, ColmapCameraExtrinsic]:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasBinary(const std::string& path)
            void Reconstruction::ReadCamerasBinary(const std::string& path)
        """
        camera_extrinsics = {}
        with open(bin_file_path, "rb") as fid:
            num_reg_images = colmap_binary_read_next_bytes(
                fid=fid,
                num_bytes=8,
                format_char_sequence="Q")[0]
            for _ in range(num_reg_images):
                binary_image_properties = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=64,
                    format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_file_name = ""
                current_char = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=1,
                    format_char_sequence="c")[0]
                while current_char != b"\x00":  # look for the ASCII 0 entry
                    image_file_name += current_char.decode("utf-8")
                    current_char = colmap_binary_read_next_bytes(
                        fid=fid,
                        num_bytes=1,
                        format_char_sequence="c")[0]
                num_pts = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=8,
                    format_char_sequence="Q")[0]
                pts_infos = colmap_binary_read_next_bytes(
                    fid=fid,
                    num_bytes=(24 * num_pts),
                    format_char_sequence=("ddq" * num_pts))
                pts2d = np.column_stack([tuple(map(float, pts_infos[0::3])), tuple(map(float, pts_infos[1::3]))])
                pts3d_ids = np.array(tuple(map(int, pts_infos[2::3])))
                camera_extrinsics[image_id] = ColmapCameraExtrinsic(
                    image_id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    image_file_name=image_file_name,
                    pts2d=pts2d,
                    pts3d_ids=pts3d_ids)
        return camera_extrinsics
