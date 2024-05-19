import os
from datetime import datetime

import dto
from pipeline import MainPipeline

setup = dto.PipelineSetup(
    PATH_TO_RESULT=r"\\ZSMK-9684-001\Data\DS\test_new_prep",
    NUM_OF_CORES=5,
    MIN_TIME=datetime(year=2023, month=11, day=10, hour=0, minute=0, second=0),
    MAX_TIME=datetime(year=2023, month=11, day=12, hour=0, minute=0, second=0),
    MARK_FILTER=True,
    MARK=['Э76ХФ', 'Э90ХАФ', '76ХФ', '90ХАФ'],
    PATH_TO_METADATA=os.path.join(
        r'\\ZSMK-9684-001\Data\DS', "metadata/debug/*xlsx"
    ),
    PATH_TO_MATERIALS=r"agregator\run\materials\*",
    METADATA_BILLET_ID="BilletId"
)


def run_aggregation():
    pipeline = MainPipeline(setup)
    pipeline.run_pipeline()


if __name__ == "__main__":
    run_aggregation()
