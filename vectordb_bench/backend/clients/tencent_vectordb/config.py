from numpy import uint64
from pydantic import BaseModel
from ..api import DBCaseConfig, DBConfig, MetricType
from ..api import IndexType as ApiIndexType
from tcvectordb.model.document import Document, Filter, SearchParams
import tcvectordb
from tcvectordb.model.enum import ReadConsistency


class TencentVectorDBConfig(DBConfig):
    url: str = ""
    username: str = ""
    key: str = ""
    read_consistency: str = ReadConsistency.EVENTUAL_CONSISTENCY
    timeout: int = 30

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "username": self.username,
            "key": self.key,
            "read_consistency": self.read_consistency,
            "timeout": self.timeout,
        }
    
class TencentVectorDBIndexConfig(BaseModel):
    index_type: ApiIndexType = ApiIndexType.HNSW
    metric_type: MetricType = MetricType.IP
    replicas: int = 0
    shard: int = 2
    
class HNSWConfig(TencentVectorDBIndexConfig, DBCaseConfig):
    m: uint64 = 16
    efconstruction: uint64 = 200
    ef: uint64 = 10

    def index_param(self) -> dict:
        return {
            "index_type": self.index_type.value,
            "metric_type": self.metric_type.value,
            "m": self.m,
            "efconstruction": self.efconstruction,
        }
    
    def search_param(self) -> SearchParams:
        return SearchParams(ef=self.ef)