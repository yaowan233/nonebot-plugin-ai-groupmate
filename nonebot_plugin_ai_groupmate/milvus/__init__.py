import asyncio
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import torch
from nonebot import get_plugin_config
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    WeightedRanker,
    AsyncMilvusClient
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from transformers import AutoModel

from ..config import Config


class MilvusOperator:
    def __init__(
            self, uri: str = "http://localhost:19530", user: str = "", password: str = ""
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.semaphore = asyncio.Semaphore(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ef = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",  # Specify the model name
            device="cuda" if torch.cuda.is_available() else "cpu",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        )
        self.bge_rf = BGERerankFunction(
            model_name="BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
            device="cuda" if torch.cuda.is_available() else "cpu"  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        )
        self.clip_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
        # Put model in evaluation mode
        self.clip_model.eval()
        self.client = MilvusClient(uri, user, password)
        self.async_client = None
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
                field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024
            )
            schema.add_field(field_name="created_at", datatype=DataType.INT64)
            # Prepare index parameters
            index_params = self.client.prepare_index_params()

            # Add indexes
            index_params.add_index(
                field_name="dense",
                index_name="dense_index",
                index_type="AUTOINDEX",
                metric_type="IP"
            )

            self.client.create_collection(
                collection_name="media_collection",
                schema=schema,
                index_params=index_params,
            )

    def _get_async_client(self) -> AsyncMilvusClient:
        """确保在当前运行的 loop 中获取或创建 client"""
        # 如果 client 不存在，或者属于旧的 loop（虽然 Python 对象本身不绑定 loop，但内部 gRPC channel 绑定）
        # 这里的简单做法是每次检测，或者为了安全，每次请求都创建一个新的（如果连接开销可接受）
        # 对于生产环境，建议使用连接池或检查当前的 loop 上下文
        if self.async_client is None:
            self.async_client = AsyncMilvusClient(self.uri, self.user, self.password)
        return self.async_client

    async def insert(self, text, session_id, collection_name="chat_collection"):
        client = self._get_async_client()
        async with self.semaphore:
            encoded = self.ef.encode_documents([text])
        dense_vector = encoded["dense"][0]
        data = {
            "session_id": session_id,
            "text": text,
            "dense": dense_vector,
            "created_at": int(time.time() * 1000),
        }
        res = await client.insert(collection_name=collection_name, data=data)
        return res

    async def batch_insert(self, texts, session_id, collection_name="chat_collection"):
        """批量插入向量"""
        if not texts:
            return []

        # 批量编码所有文本
        async with self.semaphore:
            encoded = self.ef.encode_documents(texts)
        dense_vectors = encoded["dense"]

        # 准备批量插入数据
        batch_data = []
        current_time = int(time.time() * 1000)

        for i, text in enumerate(texts):
            # 处理稀疏向量
            # 构建数据项
            data_item = {
                "session_id": session_id,
                "text": text,
                "dense": dense_vectors[i],
                "created_at": current_time,
            }
            batch_data.append(data_item)

        # 批量插入到Milvus
        client = self._get_async_client()
        res = await client.insert(collection_name=collection_name, data=batch_data)
        return res

    async def insert_media(self, media_id, image_urls, collection_name="media_collection"):
        async with self.semaphore:
            image_embeddings = await asyncio.to_thread(self.clip_model.encode_image, image_urls)
        dense_vector = image_embeddings[0]
        data = {
            "id": media_id,
            "dense": dense_vector,
            "created_at": int(time.time() * 1000),
        }
        client = self._get_async_client()
        res = await client.insert(collection_name=collection_name, data=data)
        return res

    async def search(
            self,
            text: list[str],
            search_filter: str | None = None,
            collection_name="chat_collection",
    ):
        async with self.semaphore:
            encoded = await asyncio.to_thread(self.ef.encode_documents, text)
        dense_vector = encoded["dense"][0]

        search_param_1 = {
            "data": [dense_vector],
            "anns_field": "dense",
            "param": {"nprobe": 10},
            "limit": 10,
            "expr": search_filter,
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": text,
            "anns_field": "sparse",
            "param": {"drop_ratio_search": 0.2},
            "limit": 10,
            "expr": search_filter,
        }
        request_2 = AnnSearchRequest(**search_param_2)
        reqs = [request_1, request_2]

        client = self._get_async_client()
        res = await client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=self.ranker,
            limit=10,
        )

        # 快速检查结果，避免后续空列表报错
        if not res or not res[0]:
            return None

        ids = [i["id"] for i in res[0]]

        texts = await client.get(
            collection_name=collection_name,
            ids=ids,
            output_fields=["text"]
        )


        text_list = [i["text"] for i in texts]

        async with self.semaphore:
            results = await asyncio.to_thread(self.bge_rf, text[0], text_list)

        if not results:
            return None

        best_texts = [i.text for i in results]
        return best_texts

    async def search_media(self, text):
        async with self.semaphore:
            text_embeddings = await asyncio.to_thread(self.clip_model.encode_text, text)
        dense_vector = text_embeddings[0]
        client = self._get_async_client()
        res = await client.search(
            collection_name="media_collection",
            anns_field="dense",
            data=[dense_vector],
            limit=3,
            search_params={"metric_type": "IP"}
        )
        return [i["id"] for i in res[0]]


plugin_config = get_plugin_config(Config)

MilvusOP = MilvusOperator(plugin_config.milvus_uri, plugin_config.milvus_user, plugin_config.milvus_password)
