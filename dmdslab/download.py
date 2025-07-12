from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataLoader(ABC):
    def __init__(self, connect, **kwargs):
        self.connect = connect
        self.id_column = None

    @abstractmethod
    def download(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError


class SQLLoader(DataLoader):
    def __init__(self, connect=None, query: Optional[str] = None, **kwargs):
        super().__init__(connect, **kwargs)
        self.query = query
        self.template_field = kwargs.get("template_field", None)
        self.partitions = kwargs.get("partitions", [])

    def make_queries(self):
        result = []
        if self.template_field and self.partitions:
            for partition in self.partitions:
                replace_part = (
                    f"\nAND ({self.template_field} > {partition[0]} OR {partition[0]} IS NULL)"
                    + f"\nAND ({self.template_field} <= {partition[1]} OR {partition[1]} IS NULL)"
                )
                result.append(self.query.replace("{}", replace_part))
        else:
            result.append(self.query)
        return result

    def download(self, **kwargs) -> pd.DataFrame:
        results = []
        for counter, q in enumerate(self.make_queries(), start=1):
            print(f"Partition {counter} of {len(self.partitions)}")
            print(f"Query:\n{q}")
            t_data = pd.read_sql(q, self.connect)
            print(type(t_data))
            print(t_data)
            results.append(t_data)
        result = pd.concat(results, ignore_index=True)
        if self.id_column:
            result = result.set_index(self.id_column)
        return result
