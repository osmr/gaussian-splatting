from typing import BinaryIO
import struct


def colmap_binary_read_next_bytes(fid: BinaryIO,
                                  num_bytes: int,
                                  format_char_sequence: str,
                                  endian_character: str = "<"):
    """
    Read and unpack the next bytes from a binary file.

    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
