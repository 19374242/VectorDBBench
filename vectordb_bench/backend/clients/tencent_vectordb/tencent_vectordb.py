import time
from typing import Iterable
from ..api import VectorDB
from vectordb_bench.backend.clients.tencent_vectordb.config import TencentVectorDBIndexConfig
import tcvectordb
import logging
log = logging.getLogger(__name__)
from tcvectordb.model.enum import FieldType, IndexType, MetricType
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams, SparseVector, SparseIndex
from tcvectordb.model.collection import Collection
from tcvectordb.model.database import Database
from contextlib import contextmanager
from tcvectordb.model.document import Document, Filter

class TencentVectorDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TencentVectorDBIndexConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.collection_name = collection_name
        self.drop_old = drop_old
        self.db_config = db_config
        self.client = tcvectordb.VectorDBClient(
            url=db_config["url"],
            username=db_config["username"],
            key=db_config["key"],
            read_consistency=db_config["read_consistency"],
            timeout=db_config["timeout"],
        )
        self.db_case_config = db_case_config

        self._primary_field = "id"
        self._scalar_field = "scalar_field"
        self._vector_field = "vector"
        self.db_name = "VectorDBBenchDB"
        self.batch_size = 100
        self.index = None

        if drop_old and self.client.exists_db(database_name=self.db_name):
            log.info(f"client drop_old db_name: {self.db_name}")
            self.client.drop_database(database_name=self.db_name)

        db = self.client.create_database_if_not_exists(database_name=self.db_name)

        if self.db_case_config.index_type == IndexType.HNSW:
            index = Index(
                FilterIndex(name=self._primary_field, field_type=FieldType.Uint64, index_type=IndexType.PRIMARY_KEY),
                FilterIndex(name=self._scalar_field, field_type=FieldType.Uint64, index_type=IndexType.FILTER),
                VectorIndex(name=self._vector_field, dimension=dim, index_type=self.db_case_config.index_type,
                            metric_type=self.db_case_config.metric_type, params=HNSWParams(m=self.db_case_config.m, efconstruction=self.db_case_config.efconstruction))

            ) 

        log.info(f"create collection: {self.collection_name}")
        db.create_collection(
            name=self.collection_name,
            shard=self.db_case_config.shard,
            replicas=self.db_case_config.replicas,
            index=index
        )
    
    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.db = self.client.database(self.db_name)
        self.col = self.db.describe_collection(self.collection_name)
        yield

    def optimize(self):
        assert self.db, "Please call self.init() before"
        assert self.col, "Please call self.init() before"
        
        while True:
            if self.col.index_status["status"] == "ready":
                break
            time.sleep(1)
            self.col = self.db.describe_collection(self.collection_name)
            log.info(f"col status: {self.index.stat}")
        log.info(f"optimizing before search")

    def ready_to_load(self):
        assert self.db, "Please call self.init() before"
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
                documents = []
                log.info(metadata[0])
                for i in range(batch_start_offset, batch_end_offset):
                    document = Document(id=metadata[i], vector=embeddings[i], scalar_field=metadata[i])
                    documents.append(document)
                self.client.upsert(
                    database_name=self.db_name,
                    collection_name=self.collection_name,
                    documents=documents,
                    build_index=True
                )
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
        # Perform the search.
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
            convert_filter = self._scalar_field + symble + statement.replace(symble, "").strip()
        res = self.client.search(
            database_name=self.db_name,
            collection_name=self.collection_name,
            vectors=[query],
            # filter=filters,
            params=self.db_case_config.search_param(),
            limit=k,
            filter=convert_filter
        )
        # Organize results.
        ret = [result[self._primary_field] for result in res[0]]
        # log.info(f"search result: {ret}")
        return ret