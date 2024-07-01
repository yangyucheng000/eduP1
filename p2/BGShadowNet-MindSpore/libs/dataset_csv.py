import dataclasses
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["DATASET_CSVS"]


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    val: str
    test: str


DATASET_CSVS = {
    "Jung": DatasetCSV(
        val="./csv/Jung/val.csv",
        test="./csv/Jung/test.csv",
    ),
}
