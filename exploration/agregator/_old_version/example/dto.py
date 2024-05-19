class PreprocessResult:

    def __init__(self, name: str, to_rename: bool, to_drop: bool, outliers: list,
                 count_outliers: int, count_unique: int):
        self.name = name
        self.to_rename = to_rename
        self.to_drop = to_drop
        self.outliers = outliers
        self.count_outliers = count_outliers
        self.count_unique = count_unique

    def __repr__(self):
        return f"{self.name}: {self.count_outliers} outliers, {self.count_unique} unique values \n" \
               f"to delete: {'yes' if self.to_drop else 'no'}, to rename: {'yes' if self.to_rename else 'no'}"
