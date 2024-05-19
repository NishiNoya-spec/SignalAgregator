import os
from pathlib import Path

PATH_TO_DATA = r"path/to/your/data"


def separate_rollings():

    for path in os.listdir(PATH_TO_DATA):
        if path[-4:] == ".csv":
            roll_number = path.split("_")[-2]
            path_to_rolling = os.path.join(f"{PATH_TO_DATA}_{roll_number}", "csv")

            Path(path_to_rolling).mkdir(parents=True, exist_ok=True)

            os.rename(
                os.path.join(PATH_TO_DATA, path),
                os.path.join(path_to_rolling, path),
            )


if __name__ == "__main__":
    separate_rollings()