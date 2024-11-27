from contextlib import contextmanager
import logging
import time
from typing import Iterable
from ..api import VectorDB
from volcengine.viking_db import VikingDBService, Field, FieldType, VectorIndexParams, Data, Field, IndexType

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
        self.collection_name = collection_name
        self.drop_old = drop_old
        self.db_config = db_config
        self.vikingdb_service = VikingDBService(db_config["host"], db_config["region"], db_config["ak"], db_config["sk"], db_config["scheme"])
        self.db_case_config = db_case_config

        self._primary_field = "pk"
        self._scalar_field = "id"
        self._vector_field = "vector"
        self._index_name = "vector_idx"
        self.batch_size = 100
        self.index = None

        if drop_old:
            log.info(f"client drop_old collection: {self.collection_name}")
            try:
                self.vikingdb_service.drop_collection(collection_name)
            except Exception as e:
                log.warning(f"drop_old collection: {self.collection_name} failed: {e}, skip")
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

        if self.db_case_config.index_type == IndexType.HNSW:
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

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.col = self.vikingdb_service.get_collection(self.collection_name)
        self.index = self.vikingdb_service.get_index(self.collection_name, self._index_name)
        yield

    def optimize(self):
        assert self.col, "Please call self.init() before"
        assert self.index, "Please call self.init() before"
        while True:
            if self.index.stat == "READY":
                break
            time.sleep(1)
            self.index = self.vikingdb_service.get_index(self.collection_name, self._index_name)
            log.info(f"index status: {self.index.stat}")
        log.info(f"optimizing before search")

    def ready_to_load(self):
        assert self.col, "Please call self.init() before"
        log.info(f"load")

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        assert self.col is not None
        # metadata是一个id，唯一标识一个向量
        assert len(embeddings) == len(metadata)
        insert_count = 0
        log.info(f"need to inserting {len(embeddings)} embeddings")
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                datas = []
                log.info(metadata[0])
                for i in range(batch_start_offset, batch_end_offset):
                    data = Data({
                        self._primary_field: metadata[i],
                        self._scalar_field: metadata[i],
                        self._vector_field: embeddings[i]
                    })
                    datas.append(data)
                self.col.upsert_data(datas)
                insert_count += batch_end_offset-batch_start_offset
                log.info(f"inserted {batch_end_offset} embeddings, remianing {len(embeddings)-batch_end_offset} embeddings")
        except Exception as e:
            log.info(f"Failed to insert data: {e}")
            return (insert_count, e)
        return (insert_count, None)
    
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        assert self.col is not None
        assert self.index is not None

        # Perform the search.
        res = self.index.search_by_vector(
            vector=query,
            limit=k,
            filter=filters
        )

        # Organize results.
        ret = [result.id for result in res]
        return ret