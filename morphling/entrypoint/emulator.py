import argparse
import json
import os
from collections import Counter
from multiprocessing import shared_memory

import numpy as np

from morphling.runtime import EmulationEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)

    args = parser.parse_args()

    param_meta_map_file = os.path.join(args.ckpt_path, "param_meta_map.json")

    with open(param_meta_map_file, "r") as f:
        param_meta_map = json.load(f)

    shm_mem_size, shm_mem_offsets = EmulationEngine.compute_shm_offsets(param_meta_map)

    unique_sizes_counter = Counter(
        [param["size"] for param in param_meta_map.values()]
    )

    shm_param_buffer = shared_memory.SharedMemory(create=True, size=shm_mem_size)
    buffer = shm_param_buffer.buf
    buffer[:] = bytearray(shm_mem_size)

    for size, count in unique_sizes_counter.items():
        # find all tensor name and id with the same size
        ids_of_size = EmulationEngine.find_tensor_same_size(param_meta_map, size)

        # print(f"Size: {size}, Count: {count}, Names: {names_of_size}, IDs: {ids_of_size}")

        assert (
            len(ids_of_size) == count
        ), f"Size mismatch: {len(ids_of_size)} != {count}"

        # write ids_of_size to buffer
        tmp = np.ndarray(
            ids_of_size.shape,
            dtype=ids_of_size.dtype,
            buffer=buffer,
            offset=shm_mem_offsets[size],
        )
        tmp[:] = ids_of_size[:]

        tmp = np.ndarray(
            ids_of_size.shape,
            dtype=ids_of_size.dtype,
            buffer=buffer,
            offset=shm_mem_offsets[size] + size - count * 4,
        )
        tmp[:] = ids_of_size[:]

    # print buffer as hex and skip bytes with all zeros
    print("Buffer:")
    for i in range(0, len(buffer), 16):
        line = buffer[i : i + 16]
        if line != b"\x00" * 16:
            print(f"{i:04x} {line.hex()}")

if __name__ == "__main__":
    main()
