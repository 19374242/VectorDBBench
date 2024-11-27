from pydantic import BaseModel
from ..api import DBCaseConfig, DBConfig, MetricType
from ..api import IndexType as ApiIndexType

from volcengine.viking_db import IndexType, DistanceType, QuantType

class VikingDBConfig(DBConfig):
    ak: str = ""
    sk: str = ""
    host: str = ""
    region: str = ""
    scheme: str = ""

    def to_dict(self) -> dict:
        return {
            "ak": self.ak,
            "sk": self.sk,
            "host": self.host,
            "region": self.region,
            "scheme": self.scheme,
        }
    
class VikingDBIndexConfig(BaseModel):
    index_type: ApiIndexType = ApiIndexType.HNSW
    metric_type: MetricType = MetricType.IP
    quant: QuantType = QuantType.Int8
    cpu_quota: int = 2
    scalar_index: list = None

class AutoIndexConfig(VikingDBIndexConfig, DBCaseConfig):
    def index_param(self) -> dict:
        return {
            "index_type": self.index_type.value,
            "metric_type": self.metric_type.value,
            "quant": self.quant.value,
            "cpu_quota": self.cpu_quota,
            "scalar_index": self.scalar_index,
        }
    
    def search_param(self) -> dict:
        return {}
    
class HNSWConfig(VikingDBIndexConfig, DBCaseConfig):
    index_type: ApiIndexType = ApiIndexType.HNSW
    hnsw_m: int = 20
    hnsw_cef: int = 400
    hnsw_sef: int = 800

    def index_param(self) -> dict:
        return {
            "index_type": self.index_type.value,
            "metric_type": self.metric_type.value,
            "quant": self.quant.value,
            "cpu_quota": self.cpu_quota,
            "scalar_index": self.scalar_index,
            "hnsw_m": self.hnsw_m,
            "hnsw_cef": self.hnsw_cef,
            "hnsw_sef": self.hnsw_sef,
        }
    
    def search_param(self) -> dict:
        return {}
    
_vikingdb_case_config = {
    ApiIndexType.AUTOINDEX: AutoIndexConfig,
    ApiIndexType.HNSW: HNSWConfig,
}

