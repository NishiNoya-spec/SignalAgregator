from glob import glob

import pandas as pd
from tqdm import tqdm


class TorsionGetter:

    def __init__(self, path_to_processed_data):
        self.data = self._get_data(path_to_processed_data)

    def _get_data(self, path_to_processed_data):
        data = []
        for file_path in tqdm(glob(path_to_processed_data)):
            df = pd.read_csv(file_path).rename(
                columns={"billet_id": 'BilletId'}
            )
            data.append(df)
        return pd.concat(data)
