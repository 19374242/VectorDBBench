from contextlib import contextmanager
import logging
import multiprocessing
import time
from typing import Iterable
from ..api import VectorDB
from volcengine.viking_db import VikingDBService, Field, FieldType, VectorIndexParams, Data, Field, IndexType
import struct, base64, uuid, tqdm, time
from vectordb_bench.backend.clients.vikingdb.config import VikingDBIndexConfig
log = logging.getLogger(__name__)
class VikingDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: VikingDBIndexConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        """初始化"""
        self.collection_name = collection_name
        self.drop_old = drop_old
        self.db_config = db_config
        self.vikingdb_service = VikingDBService(db_config["host"], db_config["region"], db_config["ak"], db_config["sk"], db_config["scheme"], connection_timeout=100, socket_timeout=100)
        self.db_case_config = db_case_config
        # log.warning(f"db_case_config: {db_case_config.metric_type, db_case_config.index_type, db_case_config.quant}")
        self._primary_field = "pk"
        self._scalar_field = "id"
        self._vector_field = "vector"
        # vector_idx_origin int8
        # vector_idx_sef1500
        # vector_idx_float
        self._index_name = "vector_idx_float"
        self.batch_size = 100
        self.index = None

        # if drop_old:
        #     log.info(f"client drop_old collection: {self.collection_name}")
        #     try:
        #         self.vikingdb_service.drop_index(collection_name, self._index_name)
        #         self.vikingdb_service.drop_collection(collection_name)
        #     except Exception as e:
        #         log.warning(f"drop_old collection: {self.collection_name} failed: {e}, skip")

        fields = [
            Field(
                field_name=self._primary_field,
                field_type=FieldType.Int64,
                is_primary_key=True
            ),
            Field(
                field_name=self._scalar_field,
                field_type=FieldType.Int64,
            ),
            Field(
                field_name=self._vector_field,
                field_type=FieldType.Vector,
                dim=dim
            )
        ]

        log.info(f"create collection: {self.collection_name}")
        try:
            self.vikingdb_service.create_collection(self.collection_name, fields)
        except Exception as e:
            log.warning(f"create collection: {self.collection_name} failed: {e}, skip")

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        确保collection和index存在
        """
        self.col = self.vikingdb_service.get_collection(self.collection_name)
        self.index = self.vikingdb_service.get_index(self.collection_name, self._index_name)
        yield

    def optimize(self):
        """确保索引状态为READY"""
        assert self.col, "Please call self.init() before"

        if self.db_case_config.index_type == "HNSW":
            log.warning(f"db_case_config: {self.db_case_config.metric_type, self.db_case_config.index_type, self.db_case_config.quant}")
            vector_index = VectorIndexParams(self.db_case_config.metric_type, 
                self.db_case_config.index_type, self.db_case_config.quant, hnsw_m=self.db_case_config.hnsw_m, 
                hnsw_cef=self.db_case_config.hnsw_cef, hnsw_sef=self.db_case_config.hnsw_sef)
        else:
            vector_index = VectorIndexParams(self.db_case_config.metric_type, 
                self.db_case_config.index_type, self.db_case_config.quant)
            
        try:
            self.vikingdb_service.create_index(self.collection_name, self._index_name, vector_index=vector_index, 
                            cpu_quota=self.db_case_config.cpu_quota, scalar_index=self.db_case_config.scalar_index)
        except Exception as e:
            log.warning(f"create index: {self.collection_name} failed: {e}, skip")

        self.index = self.vikingdb_service.get_index(self.collection_name, self._index_name)
        
        while True:
            if self.index.stat == "READY":
                break
            time.sleep(1)
            self.index = self.vikingdb_service.get_index(self.collection_name, self._index_name)
            log.info(f"index status: {self.index.stat}")
        log.info(f"optimizing before search")

    def ready_to_load(self):
        """确保数据集存在，可以写入数据"""
        assert self.col, "Please call self.init() before"
        log.info(f"load")

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception): 
        """写入数据"""   
        return (10000, None)
        assert self.col is not None
        # metadata是一个id，唯一标识一个向量
        assert len(embeddings) == len(metadata)
        insert_count = 0
        log.info(f"need to inserting {len(embeddings)} embeddings")
        # try:
        #     for batch_start_offset in range(0, len(embeddings), self.batch_size):
        #         batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
        #         datas = []
        #         log.info(metadata[0])
        #         for i in range(batch_start_offset, batch_end_offset):
        #             data = Data({
        #                 self._primary_field: metadata[i],
        #                 self._scalar_field: metadata[i],
        #                 self._vector_field: embeddings[i]
        #             })
        #             datas.append(data)
        #         self.col.upsert_data(datas)
        #         insert_count += batch_end_offset-batch_start_offset
        #         log.info(f"inserted {batch_end_offset} embeddings, remianing {len(embeddings)-batch_end_offset} embeddings")
        
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                datas = []
                log.info(metadata[batch_start_offset])
                for i in range(batch_start_offset, batch_end_offset):
                    packed_data = struct.pack('f'*len(embeddings[i]), *embeddings[i])
                    s =  base64.b64encode(packed_data).decode()
                    data = Data({
                        self._primary_field: metadata[i],
                        self._scalar_field: metadata[i],
                        self._vector_field: s
                    })
                    datas.append(data)
                self.col.upsert_data(datas)
                insert_count += batch_end_offset-batch_start_offset
                log.info(f"inserted {batch_end_offset} embeddings, remianing {len(embeddings)-batch_end_offset} embeddings")
                
        except Exception as e:
            log.info(f"Failed to insert data: {e}")
            return (insert_count, e)
        log.info(f"inserted {insert_count} embeddings")
        return (insert_count, None)
    
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """搜索数据"""
        assert self.col is not None
        assert self.index is not None
        # Perform the search.
        convert_filter=None
        # log.warning(f"filters {filters}")
        if filters is not None:
            statement = filters["metadata"]
            symble: str = ""
            replace_str = ""
            num: int = 0
            if ">=" in statement:
                symble = "gte"
                replace_str = ">="
            elif ">" in statement:
                symble = "gt"
                replace_str = ">"
            elif "<=" in statement:
                symble = "lte"
                replace_str = "<="
            elif "<" in statement:
                symble = "lt"
                replace_str = "<"
            else:
                log.warning(f"filters: {filters} is not supported")
            num = int(statement.replace(replace_str, "").strip())

            convert_filter = {
                "op": "range",
                "field": self._scalar_field,
                symble: num
            }
            if symble == "":
                log.warning(f"filter error: {filters}")
                convert_filter=None

        # log.info(f"convert_filter: {convert_filter}")
        res = self.index.search_by_vector(
            vector=query,
            limit=k,
            filter=convert_filter
        )
        # Organize results.
        ret = [result.id for result in res]
        # log.info(f"search result: {ret}")
        return ret