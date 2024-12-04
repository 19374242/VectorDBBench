from contextlib import contextmanager
import logging
import multiprocessing
import time
from typing import Iterable

import dashvector
from ..api import VectorDB
import struct, base64, uuid, tqdm, time
from vectordb_bench.backend.clients.dash_vector.config import DashVectorIndexConfig
log = logging.getLogger(__name__)
from dashvector import Doc, VectorParam, VectorQuery

class DashVector(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DashVectorIndexConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.collection_name = collection_name
        self.drop_old = drop_old
        self.db_config = db_config
        self.client = dashvector.Client(
            api_key=db_config["api_key"],
            endpoint=db_config["endpoint"]
        )
        self.db_case_config = db_case_config

        self._primary_field = "id"
        self._scalar_field = "scalar_field"
        self._vector_field = "vector"
        self._index_name = "vector_idx"
        self.batch_size = 100
        self.index = None

        if drop_old:
            log.info(f"client drop_old collection: {self.collection_name}")
            try:
                self.client.delete(collection_name=self.collection_name)
            except Exception as e:
                log.warning(f"drop_old collection: {self.collection_name} failed: {e}, skip")

        ret = self.client.create(
            name=self.collection_name,
            dimension=dim,
            metric=self.db_case_config.metric_type,
            dtype=float,
            fields_schema={self._primary_field: int, self._scalar_field: int},
            vectors=VectorParam(
                dimension=dim,
                dtype=float,
                metric=self.db_case_config.metric_type,
                quantize_type=self.db_case_config.quant
            )
        )

        if ret:
            log.info('create collection success!')
        else:
            log.error(f'create collection failed: {ret}')

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.col = self.client.get(self.collection_name)
        yield

    def optimize(self):
        assert self.col, "Please call self.init() before"
        ret = self.client.describe(self.collection_name)
        
        while True:
            if ret.output["status"] == "SERVING":
                break
            time.sleep(1)
            ret = self.client.describe(self.collection_name)
            log.info(f"index status: {ret.status}")
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
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                docs = []
                for i in range(batch_start_offset, batch_end_offset):
                    doc = Doc(
                        id=metadata[i],
                        fields={
                            self._primary_field: metadata[i],
                            self._scalar_field: metadata[i],
                        }
                    )
                res = self.col.insert(docs=docs)
                insert_count += len(res.primary_keys)
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
        convert_filter=None
        if filters is not None:
            statement = filters["metadata"]
            symble: str = ""
            if ">=" in statement:
                symble = ">="
            elif ">" in statement:
                symble = ">"
            elif "<=" in statement:
                symble = "<="
            elif "<" in statement:
                symble = "<"
            else:
                log.warning(f"filters: {filters} is not supported")
            convert_filter = self._scalar_field + " " + symble + " " + statement.replace(symble, "").strip()

        # Perform the search.
        res = self.col.query(
            vector=VectorQuery(vector=query, ef=self.db_case_config.search_param["ef"]),
            topk=k,
            filter=convert_filter
        )
        ret = [result.id for result in res]
        # log.info(f"search result: {ret}")
        return ret