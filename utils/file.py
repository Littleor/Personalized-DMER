import os
from typing import List


def file_list_sort(file_path_list: List[str]):
    def extract_number(filename):
        try:
            return int("".join(filter(str.isdigit, filename)))
        except ValueError:
            return float("inf")

    def custom_key(filepath):
        filename = os.path.basename(filepath)
        return (extract_number(filename), filename)

    sorted_list = sorted(file_path_list, key=custom_key)
    return sorted_list


def check_file_exist(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError("File {} not found".format(file_path))
    if os.path.isdir(file_path):
        raise IsADirectoryError("File {} is a directory".format(file_path))
    return file_path
