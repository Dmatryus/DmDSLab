from collections import namedtuple

ModelData = namedtuple("ModelData", ["features", "target"])
DataSplit = namedtuple("DataSplit", ["train", "tuning", "test"])
