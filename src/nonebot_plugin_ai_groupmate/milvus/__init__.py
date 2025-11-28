import asyncio
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from nonebot import get_plugin_config, get_driver, logger
from pymilvus import AnnSearchRequest, AsyncMilvusClient, DataType, Function, FunctionType, MilvusClient, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
import torch
from transformers import AutoModel

from ..config import Config


class MilvusOperator:
    def __init__(
            self, uri: str = "http://localhost:19530", user: str = "", password: str = ""
    ):
        # 1. __init__ 中只保存配置，不连接数据库，不加载模型
        self.uri = uri
        self.user = user
        self.password = password
        self.semaphore = asyncio.Semaphore(1)

        # 将模型和客户端占位符设为 None
        self.ef = None
        self.bge_rf = None
        self.clip_model = None
        self.client = None
        self.async_client = None
        self.ranker = None
        self.initialized = False

    async def init_models(self):
        """
        2. 创建一个专门的初始化方法，在 NoneBot 启动时调用
        """
        if self.initialized:
            return

        logger.info("正在加载 Milvus 模型和连接数据库...")
        try:
            # 这里的耗时操作只会发生在启动阶段，而不是 import 阶段
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 使用 asyncio.to_thread 避免阻塞主线程（虽然模型加载主要是 CPU/IO 密集型）
            self.ef = await asyncio.to_thread(BGEM3EmbeddingFunction,
                                              model_name="BAAI/bge-m3",
                                              device=str(device)
                                              )

            self.bge_rf = await asyncio.to_thread(BGERerankFunction,
                                                  model_name="BAAI/bge-reranker-v2-m3",
                                                  device=str(device)
                                                  )

            self.clip_model = await asyncio.to_thread(
                AutoModel.from_pretrained,
                "jinaai/jina-clip-v2",
                trust_remote_code=True
            )
            self.clip_model.to(device)
            self.clip_model.eval()

            # 初始化客户端
            self.client = MilvusClient(self.uri, self.user, self.password)
            self.ranker = WeightedRanker(0.8, 0.3)

            # 初始化 Collections (逻辑保持不变)
            self._init_collections()

            self.initialized = True
            logger.success("Milvus 模型加载及数据库连接完成。")

        except Exception as e:
            logger.error(f"Milvus 初始化失败: {e}")
            raise e

    def _init_collections(self):
        """将创建 Collection 的逻辑抽离出来"""
        if not self.client.has_collection(collection_name="chat_collection"):
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
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
        if self.async_client is None:
            self.async_client = AsyncMilvusClient(self.uri, self.user, self.password)
        return self.async_client

    async def insert(self, text, session_id, collection_name="chat_collection"):
        # 安全检查：确保初始化完成
        if not self.initialized:
            logger.warning("MilvusOperator 尚未初始化，正在尝试初始化...")
            await self.init_models()

        client = self._get_async_client()
        async with self.semaphore:
            # 此时 self.ef 已经被初始化
            encoded = await asyncio.to_thread(self.ef.encode_documents, [text])
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
        if not self.initialized:
            await self.init_models()

        if not texts:
            return []

        async with self.semaphore:
            encoded = await asyncio.to_thread(self.ef.encode_documents, texts)
        dense_vectors = encoded["dense"]

        batch_data = []
        current_time = int(time.time() * 1000)

        for i, text in enumerate(texts):
            data_item = {
                "session_id": session_id,
                "text": text,
                "dense": dense_vectors[i],
                "created_at": current_time,
            }
            batch_data.append(data_item)

        client = self._get_async_client()
        res = await client.insert(collection_name=collection_name, data=batch_data)
        return res

    async def insert_media(self, media_id, image_urls, collection_name="media_collection"):
        if not self.initialized:
            await self.init_models()

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
        if not self.initialized:
            await self.init_models()

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
        if not self.initialized:
            await self.init_models()

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

# 4. 获取驱动器并注册启动钩子
driver = get_driver()


@driver.on_startup
async def _():
    # 5. 在机器人启动时才真正下载模型、连接数据库
    await MilvusOP.init_models()