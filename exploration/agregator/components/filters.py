from source_data import Array, SourceDataset


class Filters:

    def __init__(self):
        self.methods = {
            "std": self.remove_by_std,
            "forbidden_columns": self.remove_by_forbidden_columns
        }

    def filter_by(self, method: str, *args, **kwargs):
        return self.methods[method](*args, **kwargs)

    @staticmethod
    def remove_by_std(data_array: Array, *args, **kwargs):
        if data_array.is_numeric:
            return True if data_array.values.std() == 0 else False
        else:
            return False

    @staticmethod
    def remove_by_forbidden_columns(
        data_array: Array,
        source_data: SourceDataset,
        *args,
        **kwargs,
    ):
        if (data_array.transcription.name
                in source_data.settings.forbidden_columns):
            return True
        else:
            return False
