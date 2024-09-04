import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)

    args = parser.parse_args()

    name_size_map_path = os.path.join(args.ckpt_path, "name_size_map.json")

    with open(name_size_map_path, "r") as f:
        name_size_map = json.load(f)

    unique_sizes_counter = Counter(name_size_map.values())
    print(unique_sizes_counter)

    shm_mem_size = sum([size +  4*count for size, count in unique_sizes_counter.items()]) # 4 byte uint32_t for tensor ID
    pin_mem_size = sum([size*count for size, count in unique_sizes_counter.items()])


if __name__ == "__main__":
    main()