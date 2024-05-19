from copy import copy
from typing import Dict, List


class Transcription:

    def __init__(
        self,
        workcenter: str = "",
        workunit: str = "",
        rolling_number: str = "",
        name: str = "",
        model: str = "",
        interpolation: str = "",
        sector_range: str = "",
        preprocessing: str = "",
        aggregation: str = "",
        secondary_functions: str = ""
    ):
        self.workcenter = workcenter
        self.workunit = workunit
        self.rolling_number = rolling_number
        self.name = name
        self.model = model
        self.interpolation = interpolation
        self.sector_range = sector_range
        self.preprocessing = preprocessing
        self.aggregation = aggregation
        self.secondary_functions = secondary_functions

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return (
            f"{self._add_under(self.workcenter)}"
            f"{self._add_under(self.workunit)}"
            f"{self._add_under(self.rolling_number)}"
            f"{self._add_under(self.name)}"
            f"{self._add_under(self.model)}"
            f"{self._add_under(self.interpolation)}"
            f"{self._add_under(self.sector_range)}"
            f"{self._add_dot(self.preprocessing)}"
            f"{self._add_under(self.aggregation)}"
        )[:-1]

    def add_tags(self, tags: Dict[str, List[str]], replace: bool = False):
        new_transcription = copy(self)
        for tag_type, tags_values in tags.items():
            if replace:
                full_tag = tags_values[0]
                for tag in tags_values[1:]:
                    full_tag += new_transcription._add_under(tag)
                setattr(new_transcription, tag_type, full_tag)
            else:

                for tag in tags_values:
                    transcription_attr = getattr(new_transcription, tag_type)
                    setattr(
                        new_transcription,
                        tag_type,
                        f"{new_transcription._add_under(transcription_attr)}"
                        f"{tag}",
                    )
        return new_transcription

    @staticmethod
    def _add_under(string: str):
        return f"{string}_" if string != "" else ""

    @staticmethod
    def _add_dot(string: str):
        return f"{string}." if string != "" else ""
