import math
import os
import shutil
import struct
from datetime import datetime, timedelta
from glob import glob
from multiprocessing import pool
from typing import List

import pandas as pd
import py7zr
from tqdm import tqdm

from exploration.utils.utils import number_to_str

MIN_TIME = datetime(year=2023, month=11, day=2, hour=0, minute=0, second=0)
MAX_TIME = datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0)

DATA_PATH = r'\\zsmk-fs-016.sib.evraz.com\UKR-64E'
SAVE_PATH = f"ukr64_bin_{str(MIN_TIME.date())}_{str(MAX_TIME.date())}"
TIME_FORMAT = "%H:%M:%S"    # формат времени внутри бинарного файла
DAY_FORMAT = "%d.%m.%Y"    # формат даты внутри бинарного файла
NUM_OF_CORES = 15    # число параллельных потоков программы
DATE_FORMAT = f"{DAY_FORMAT} {TIME_FORMAT}"    # итоговый формат даты_времени


class ByteReader:

    def __init__(self, file_path):
        self.file = open(file_path, 'rb')
        self.point_pos = 0

    def __del__(self):
        self.file.close()

    def seek(self, offset):
        self.point_pos = offset
        self.file.seek(offset)

    def _read(self, size):
        data = self.file.read(size)
        self.point_pos += size

        return data

    def read_text(self, size):
        return self._read(size).decode('cp1251')

    def read_uint(self, size):
        return int.from_bytes(
            self._read(size), byteorder='little', signed=False
        )

    def read_int(self, size):
        return int.from_bytes(
            self._read(size),
            byteorder='little',
            signed=True,
        )

    def read_float(self, size):
        return struct.unpack('<f', self._read(size))


class BytesDeserializer:

    def __init__(self, file_path):
        self.byte_reader = ByteReader(file_path)

    def read_header(self):
        header_size = self.read_ushort()
        header_version = self.read_ushort()

        date = self.read_string(20)
        date = datetime.strptime(date, DATE_FORMAT)

        oper_number = self.read_string(32)
        tab_number = self.read_string(30)

        _ = self.read_byte_array(46)
        billet_id = self.read_string(32)

        _ = self.read_byte_array(3)

        rail_length = self.read_int()

        factory = self.read_string(128)

        math_status = self.read_byte()

        _ = self.read_byte_array(16)

        dp_step = self.read_float()

        pass_number = self.read_byte()

        tables_count = self.read_int()
        _ = self.read_byte_array(header_size - self.byte_reader.point_pos)

        return (
            header_version, date, oper_number, tab_number, billet_id,
            rail_length, factory, math_status, dp_step, pass_number,
            tables_count
        )

    def read_ushort(self):
        return self.byte_reader.read_uint(2)

    def read_uint(self):
        return self.byte_reader.read_uint(3)

    def read_int(self):
        return self.byte_reader.read_int(4)

    def read_float(self):
        return self.byte_reader.read_float(4)

    def read_char(self):
        return self.byte_reader.read_text(1)

    def read_byte(self):
        return self.byte_reader.read_uint(1)

    def read_char_array(self, array_size):
        return [self.read_char() for _ in range(array_size)]

    def read_string(self, string_size):
        return str("".join(self.read_char_array(string_size))
                   ).replace('\x00', '')

    def read_byte_array(self, array_size):
        return [self.read_byte() for _ in range(array_size)]

    def read_channel_prop(self, j):
        return {
            f"channel_{j}": self.read_byte(),
            f"size_{j}": self.read_int(),
            f"enable_{j}": self.read_byte(),
        }

    def read_us_data(self, j, data_length):
        return {
            f"channel_{j}": self.read_ushort(),
            f"index_{j}": self.read_uint(),
            f"TOF1_{j}": self.read_float(),
            f"Ampl2_{j}": self.read_byte(),
            f"TOF2_{j}": self.read_float(),
            f"Ampl3_{j}": self.read_byte(),
            f"TOF3_{j}": self.read_float(),
            "empty_byte": self.read_byte(),
            f"data_{j}": self.read_byte_array(data_length)
        }

    def read_track_point(self, j):
        return {
            f"offset_{j}": self.byte_reader.read_uint(4),
            f"ttl_{j}": self.read_byte(),
            f"position_{j}": self.byte_reader.read_uint(4),
        }

    def read_defect_coord(self, j):
        return {
            f"amp_{j}": self.read_float(),
            f"channel_{j}": self.byte_reader.read_uint(4),
            f"offset_{j}": self.byte_reader.read_uint(4),
            f"time_{j}": self.read_float(),
            f"length_{j}": self.byte_reader.read_uint(4),
            f"type_{j}": self.byte_reader.read_uint(4),
            f"is_double_{j}": self.byte_reader.read_uint(4),
            f"cycle_{j}": self.byte_reader.read_uint(4),
        }

    def read_ttl_info(self, j):
        return {
            f"offset_{j}": self.byte_reader.read_uint(4),
            f"ttl_{j}": self.read_byte(),
        }

    def read_stamp_info(self, j):
        return {
            f"start_{j}": self.read_float(),
            f"end_{j}": self.read_float(),
        }

    def read_coupling_status(self, j):
        return {
            f"channel_{j}": self.read_int(),
            f"value_{j}": self.read_int(),
            f"status_{j}": self.read_int(),
        }


def get_data_paths() -> List[str]:
    paths = []
    for day in range((MAX_TIME - MIN_TIME).days):
        cur_date = MIN_TIME + timedelta(days=day)
        day = number_to_str(cur_date.day)
        month = number_to_str(cur_date.month)
        year = cur_date.year
        paths.extend(
            glob(
                os.path.join(
                    DATA_PATH, year, f'{day}_{month}_{cur_date.year}*.7z'
                )
            )
        )
    return paths


def data_mapping():
    cur_date = MIN_TIME
    one_day = timedelta(days=1)
    data = []
    while cur_date <= MAX_TIME and cur_date <= datetime.now():
        data_path = glob(
            os.path.join(
                r"\\ZSMK-9684-001\Data",
                str(cur_date.year),
                number_to_str(cur_date.month),
                number_to_str(cur_date.day),
                "ukr64",
                '*.csv',
            )
        )
        cur_date = cur_date + one_day
        if len(data_path) == 0:
            continue

        data.append(
            pd.read_csv(
                data_path[0],
                sep=";",
                usecols=["UniqueStringIDLnk", "IdRailUkr"],
                encoding="windows-1251"
            ).set_index("IdRailUkr")
        )

    return pd.concat(data).to_dict()["UniqueStringIDLnk"]


def get_split_data(info):
    zone = ""
    cur_table = dict()
    for data in info.split("\n"):
        split_data = data.split(" = ")
        if len(split_data) == 1:
            zone = str(data)
        elif len(split_data) == 2:
            cur_table[split_data[0] + "_" + zone] = str(split_data[1])
    return cur_table


def create_data_batch(meta_info):
    save_path = f"test{meta_info['thread_num']}"
    try:
        with py7zr.SevenZipFile(meta_info["data_path"], mode='r') as z:
            # Extract all contents to the current directory
            z.extractall(save_path)
    except py7zr.exceptions.Bad7zFile:
        print(f'error {meta_info["data_path"]}')
        return

    bin_path = glob(os.path.join(save_path, "*.ukr25"))
    if len(bin_path) == 0:
        return

    bytes_deserializer = BytesDeserializer(bin_path[0])

    (
        _,
        date,
        _,
        _,
        billet_id,
        _,
        _,
        _,
        _,
        _,
        tables_count,
    ) = bytes_deserializer.read_header()
    data_tables = dict()
    for i in range(tables_count):
        block_id = bytes_deserializer.read_int()
        data_tables[block_id] = dict()
        data_tables[block_id]['items'] = bytes_deserializer.read_int()
        data_tables[block_id]['offset'] = bytes_deserializer.read_int()
        data_tables[block_id]['length'] = bytes_deserializer.read_int()
        data_tables[block_id]['control_sum'] = (
            bytes_deserializer.read_byte_array(64)
        )
        data_tables[block_id]['info'] = []

    block_ids = list(data_tables.keys())
    cur_table = dict()
    block_actions = {
        5: bytes_deserializer.read_channel_prop,
        6: bytes_deserializer.byte_reader.seek,
        7: bytes_deserializer.read_track_point,
        8: bytes_deserializer.read_defect_coord,
        9: bytes_deserializer.read_ttl_info,
        11: bytes_deserializer.read_stamp_info,
        12: bytes_deserializer.read_coupling_status,
        101: bytes_deserializer.read_ushort,
    }
    for block_id in block_ids:
        info = data_tables[block_id]
        if block_id in block_actions.keys():
            if block_id == 6:
                block_actions[block_id](info['offset'] + info['length'])
            for j in range(info['items']):
                if block_id == 101:
                    block_info = block_actions[block_id]()
                else:
                    block_info = block_actions[block_id](j)

                data_tables[block_id]['info'].append(block_info)

        elif block_id in [1, 2, 3, 4, 10, 13]:
            data_tables[block_id]['info'] = "".join(
                bytes_deserializer.read_char_array(
                    data_tables[block_id]['length']
                )
            )

            cur_table.update(get_split_data(info["info"]))

        else:
            length = data_tables[block_id]['length']
            block_info = bytes_deserializer.read_byte_array(length)
            data_tables[block_id]['info'] = block_info

        if block_id != 5:
            del data_tables[block_id]

    cur_table["billet_id"] = billet_id
    bytes_deserializer.byte_reader.file.close()

    shutil.rmtree(save_path)

    return cur_table


if __name__ == "__main__":

    os.makedirs(SAVE_PATH, exist_ok=True)
    mapping = data_mapping()

    result_table = list()
    data_paths = get_data_paths()
    multiproc_queue = []
    for n in range(math.ceil(len(data_paths) / NUM_OF_CORES)):
        cut = data_paths[(NUM_OF_CORES * n):(NUM_OF_CORES * (n + 1))]
        multiproc_queue.append(
            [
                {
                    "thread_num": i,
                    "data_path": data_path
                } for i, data_path in enumerate(cut)
            ]
        )

    for cut_num in tqdm(range(len(multiproc_queue)),
                        bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}'):
        cut = multiproc_queue[cut_num]

        with pool.Pool(NUM_OF_CORES) as p:
            batch = p.map(create_data_batch, cut)
        result_table = pd.DataFrame(
            [element for element in batch if element is not None]
        )
        result_table["billet_id"] = result_table["billet_id"].map(mapping)
        result_table.set_index("billet_id").to_csv(
            os.path.join(SAVE_PATH, f"{cut_num}.csv")
        )
