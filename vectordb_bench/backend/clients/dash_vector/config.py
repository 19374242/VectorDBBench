import dashvector
from pydantic import BaseModel
from ..api import DBCaseConfig, DBConfig, MetricType
from ..api import IndexType as ApiIndexType


class DashVectorConfig(DBConfig):
    api_key: str = ""
    endpoint: str = ""
    protocol: dashvector.DashVectorProtocol = dashvector.DashVectorProtocol.GRPC
    timeout: float = 10.0

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key,
            "endpoint": self.endpoint,
            "protocol": self.protocol,
            "timeout": self.timeout,
        }
    
class DashVectorIndexConfig(BaseModel):
    index_type: ApiIndexType = ApiIndexType.HNSW
    metric_type: str = "dotproduct"
    quant: str = "int8"

class HNSWConfig(DashVectorIndexConfig, DBCaseConfig):
    m: int = 16
    efconstruction: int = 200
    ef: int = 10

    def index_param(self) -> dict:
        return {
            "index_type": self.index_type.value,
            "metric_type": self.metric_type.value,
            "m": self.m,
            "efconstruction": self.efconstruction,
        }
    
    def search_param(self) -> dict:
        return {
            "ef": self.ef,
        }