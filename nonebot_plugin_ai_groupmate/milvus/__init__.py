import time
from typing import Optional

from nonebot import get_plugin_config
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from ..config import Config

class MilvusOperator:
    def __init__(
        self, uri: str = "http://localhost:19530", user: str = "", password: str = ""
    ):
        self.ef = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",  # Specify the model name
            device="cpu",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=False,
        )
        self.client = MilvusClient(uri, user, password)
        self.ranker = WeightedRanker(0.8, 0.3)
        if not self.client.has_collection(collection_name="chat_collection"):
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            # Add fields to schema
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(
                field_name="session_id", datatype=DataType.VARCHAR, max_length=1000
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                enable_analyzer=True,
                max_length=20000,
            )
            # Define a sparse vector field to generate spare vectors with BM25
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(
                field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024
            )
            schema.add_field(field_name="created_at", datatype=DataType.INT64)
            bm25_function = Function(
                name="text_bm25_emb",  # Function name
                input_field_names=[
                    "text"
                ],  # Name of the VARCHAR field containing raw text data
                output_field_names=["sparse"],
                # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
                function_type=FunctionType.BM25,
            )

            schema.add_function(bm25_function)
            # Prepare index parameters
            index_params = self.client.prepare_index_params()

            # Add indexes
            index_params.add_index(
                field_name="dense",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )

            index_params.add_index(
                field_name="sparse",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
                metric_type="BM25",  # Set to `BM25` when using function to generate sparse vectors
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
                # The ratio of small vector values to be dropped during indexing
            )
            self.client.create_collection(
                collection_name="chat_collection",
                schema=schema,
                index_params=index_params,
            )
        if not self.client.has_collection(collection_name="media_collection"):
            schema = MilvusClient.create_schema(
                enable_dynamic_field=True,
            )
            # Add fields to schema
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(
                field_name="emotion", datatype=DataType.VARCHAR, max_length=100
            )
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                enable_analyzer=True,
                max_length=20000,
            )
            # Define a sparse vector field to generate spare vectors with BM25
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(
                field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024
            )
            schema.add_field(field_name="created_at", datatype=DataType.INT64)
            bm25_function = Function(
                name="text_bm25_emb",  # Function name
                input_field_names=[
                    "text"
                ],  # Name of the VARCHAR field containing raw text data
                output_field_names=["sparse"],
                # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
                function_type=FunctionType.BM25,
            )

            schema.add_function(bm25_function)
            # Prepare index parameters
            index_params = self.client.prepare_index_params()

            # Add indexes
            index_params.add_index(
                field_name="dense",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )

            index_params.add_index(
                field_name="sparse",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
                metric_type="BM25",  # Set to `BM25` when using function to generate sparse vectors
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
                # The ratio of small vector values to be dropped during indexing
            )
            self.client.create_collection(
                collection_name="media_collection",
                schema=schema,
                index_params=index_params,
            )

    def insert(self, text, session_id, collection_name="chat_collection"):
        dense_vector = self.ef.encode_documents([text])["dense"][0]
        data = {
            "session_id": session_id,
            "text": text,
            "dense": dense_vector,
            "created_at": int(time.time() * 1000),
        }
        res = self.client.insert(collection_name=collection_name, data=data)
        return res

    def insert_media(self, media_id, text, emotion, collection_name="media_collection"):
        dense_vector = self.ef.encode_documents([text])["dense"][0]
        data = {
            "id": media_id,
            "emotion": emotion,
            "text": text,
            "dense": dense_vector,
            "created_at": int(time.time() * 1000),
        }
        res = self.client.insert(collection_name=collection_name, data=data)
        return res

    def search(
        self,
        text: list[str],
        search_filter: Optional[str] = None,
        collection_name="chat_collection",
    ):
        dense_vector = self.ef(text)["dense"]
        search_param_1 = {
            "data": dense_vector,
            "anns_field": "dense",
            "param": {"metric_type": "IP", "params": {"nprobe": 10}},
            "limit": 10,
            "expr": search_filter,
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": text,
            "anns_field": "sparse",
            "param": {
                "metric_type": "BM25",
            },
            "limit": 10,
            "expr": search_filter,
        }
        request_2 = AnnSearchRequest(**search_param_2)

        reqs = [request_1, request_2]
        res = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=self.ranker,
            limit=3,
        )
        return res

    def query_ids(self, ids: list[int], collection_name="chat_collection"):
        res = self.client.get(
            collection_name=collection_name, ids=ids, output_fields=["text"]
        )
        return res


plugin_config = get_plugin_config(Config)

MilvusOP = MilvusOperator(plugin_config.milvus_uri, plugin_config.milvus_user, plugin_config.milvus_password)
