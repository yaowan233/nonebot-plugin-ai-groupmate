import os
import time
import uuid
import base64
import asyncio
import mimetypes

import httpx
from openai import AsyncOpenAI
from nonebot import get_plugin_config
from nonebot.log import logger
from qdrant_client import AsyncQdrantClient, models

from .config import Config

plugin_config = get_plugin_config(Config).ai_groupmate

class VectorDBOperator:
    def __init__(self):
        # 1. 初始化 Qdrant 客户端
        self.client = AsyncQdrantClient(
            url=plugin_config.qdrant_uri,
            api_key=plugin_config.qdrant_api_key,
            timeout=60
        )

        self.chat_col = "chat_collection"
        self.media_col = "media_collection"

        # 2. Embedding API (用于文本 -> 向量)
        # 使用硅基流动/OpenAI兼容接口
        self.emb_client = AsyncOpenAI(
            api_key=plugin_config.embedding_api_key,
            base_url=plugin_config.embedding_base_url
        )
        self.emb_model = "BAAI/bge-m3"

        # 3. Rerank API 配置
        self.rerank_url = plugin_config.rerank_api_url
        self.rerank_key = plugin_config.rerank_api_key


        self._init_lock = asyncio.Lock()
        self._collections_ready = False

    # ================= 内部工具函数 =================
    async def _ensure_collections(self):
        """
        初始化集合：如果集合不存在，则创建并开启 Int8 量化。
        """
        if self._collections_ready:
            return

        async with self._init_lock:
            if self._collections_ready:
                return

            # 1. 检查并创建 Chat 集合
            if not await self.client.collection_exists(self.chat_col):
                await self.client.create_collection(
                    collection_name=self.chat_col,
                    vectors_config=models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    ),
                )
                # 创建 session_id 索引 (加速过滤)
                await self.client.create_payload_index(
                    collection_name=self.chat_col,
                    field_name="session_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info(f"Qdrant集合 {self.chat_col} 已创建")

            # 2. 检查并创建 Media 集合
            if not await self.client.collection_exists(self.media_col):
                await self.client.create_collection(
                    collection_name=self.media_col,
                    vectors_config=models.VectorParams(
                        size=2560,
                        distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"Qdrant集合 {self.media_col} 已创建")

            self._collections_ready = True

    async def _get_text_embedding(self, text: str) -> list[float] | None:
        """调用 API 获取文本 Dense 向量 (BGE-M3)"""
        try:
            resp = await self.emb_client.embeddings.create(
                input=[text],
                model=self.emb_model
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding API Error: {e}")
            return None

    async def _get_qwen_vl_embedding(self, text: str = "", image_source: str = "") -> list[float] | None:
        """调用阿里云 Qwen3-VL-Embedding 获取多模态向量 (纯异步版)"""

        # 阿里云多模态 Embedding 的原生 REST API 地址
        aliyun_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

        headers = {
            "Authorization": f"Bearer {plugin_config.qwen_token}",
            "Content-Type": "application/json"
        }

        item = {}

        # 1. 填充文本（如果只有文本，就是纯文本搜索）
        if text:
            item["text"] = text

        # 2. 填充图片（支持本地文件路径、纯 Base64、带有 Header 的 Base64 以及 http 链接）
        if image_source:
            # 场景 A: 传入的是本地文件路径 (如 "/path/to/meme.png")
            if os.path.isfile(image_source):
                # 自动推断图片的 mime_type
                mime_type, _ = mimetypes.guess_type(image_source)
                mime_type = mime_type or "image/jpeg"  # 兜底

                with open(image_source, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                    # ⚠️ 阿里要求必须拼装上 data:image/... 的头部
                    item["image"] = f"data:{mime_type};base64,{base64_data}"

            # 场景 B: 传入的已经是标准的 Data URI (前端传来的 data:image/png;base64,xxx...)
            elif image_source.startswith("data:image"):
                # ⚠️ 和 Jina 最大的不同：千万【不要】切掉头部，直接原样传给阿里
                item["image"] = image_source

            # 场景 C: 普通网络图片
            elif image_source.startswith("http://") or image_source.startswith("https://"):
                item["image"] = image_source

            # 场景 D: 传入的是被切掉头部的纯 Base64 字符串 (兜底处理)
            else:
                # 假设它是纯 Base64，给它强行补上头部
                item["image"] = f"data:image/jpeg;base64,{image_source}"

        if not item:
            logger.warning("Aliyun Embedding: text 和 image_source 均为 None")
            return None

        # 3. 构造最终 Payload
        # 如果 item 里同时包含了 "text" 和 "image"，阿里会自动进行多模态融合！
        payload = {
            "model": "qwen3-vl-embedding",
            "input": {
                "contents": [item]
            }
        }

        max_retries = 3  # 最多试3次

        # client 在循环外创建，避免每次 retry 都重新握手
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(aliyun_url, json=payload, headers=headers)

                    if resp.status_code != 200:
                        logger.error(f"Aliyun API Error {resp.status_code}: {resp.text}")
                        # 4xx 错误通常是参数错误或欠费，没必要重试
                        if 400 <= resp.status_code < 500:
                            return None
                        resp.raise_for_status()

                    # ⚠️ 解析阿里的响应结构
                    # 阿里的成功返回 JSON 是: {"output": {"embeddings":[{"embedding": [0.1, 0.2...], "text_index": 0}]}}
                    return resp.json()["output"]["embeddings"][0]["embedding"]

                except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout) as e:
                    is_last_attempt = (attempt == max_retries - 1)

                    if is_last_attempt:
                        logger.error(f"Aliyun API 重试3次后最终失败: {repr(e)}")
                        return None
                    else:
                        wait_time = 2 * (attempt + 1)  # 2秒, 4秒...
                        logger.warning(f"Aliyun API 连接抖动 ({repr(e)})，正在第 {attempt + 1} 次重试...")
                        await asyncio.sleep(wait_time)

                except Exception as e:
                    logger.error(f"Aliyun API 未知异常: {e}")
                    return None

        return None


    async def _rerank(self, query: str, docs: list[str]) -> list[str]:
        """调用 Rerank API 对结果精排"""
        if not docs:
            return []

        # 如果只有一条，没必要 Rerank
        if len(docs) == 1:
            return docs

        try:
            headers = {
                "Authorization": f"Bearer {self.rerank_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": docs,
                "top_n": 5  # 只取前5最相关的
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.rerank_url, json=payload, headers=headers)
                resp.raise_for_status()
                results = resp.json().get("results", [])

                # 按相关性分数排序
                results.sort(key=lambda x: x["relevance_score"], reverse=True)

                # 返回排序后的文本
                return [docs[item["index"]] for item in results]
        except Exception as e:
            logger.error(f"Rerank API Error: {e}")
            # 降级策略：如果 Rerank 挂了，直接返回前 5 条
            return docs[:5]

    # ================= 聊天记录功能 (RAG) =================

    async def insert_chat(self, text: str, session_id: str):
        """插入新的聊天记录"""
        await self._ensure_collections()
        vector = await self._get_text_embedding(text)
        if not vector:
            return

        point_id = str(uuid.uuid4())

        await self.client.upsert(
            collection_name=self.chat_col,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "text": text,
                        "created_at": int(time.time())
                    }
                )
            ]
        )

    async def search_chat(self, query: str, session_id: str) -> str:
        """
        RAG 搜索核心逻辑 (适配 query_points 接口)
        """
        await self._ensure_collections()
        # 1. 获取向量
        vector = await self._get_text_embedding(query)
        if not vector:
            return "无法连接记忆库"

        # 2. Qdrant 向量搜索
        # 使用 query_points() 接口
        search_result = await self.client.query_points(
            collection_name=self.chat_col,
            query=vector,               # <--- 对应文档: If list[float] - use as dense vector
            query_filter=models.Filter( # <--- 对应文档: 参数名是 query_filter
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            ),
            limit=20
        )

        # 注意：query_points 返回的是 QueryResponse
        # 它的结构通常包含 points 列表
        if not search_result or not search_result.points:
            return "未找到相关历史记录"

        # 提取文本内容
        # search_result.points 是 ScoredPoint 的列表
        candidates = [point.payload["text"] for point in search_result.points if point.payload and "text" in point.payload]

        # 3. Rerank 重排序
        best_texts = await self._rerank(query, candidates)

        return "\n".join(best_texts)

    # ================= 表情包功能 (Image Search) =================

    async def insert_media(self, media_id: int, image_url: str, description: str) -> None:
        """插入新表情包 (新图入库用)"""
        await self._ensure_collections()
        vector = await self._get_qwen_vl_embedding(image_source=image_url, text=description)
        if not vector:
            return

        await self.client.upsert(
            collection_name=self.media_col,
            points=[
                models.PointStruct(
                    id=media_id,  # 保持 Int ID
                    vector=vector,
                    payload={
                        "created_at": int(time.time())
                    }
                )
            ]
        )

    async def _get_batch_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        批量调用 API 获取文本向量 (自动处理 Batch Size 限制)
        """
        if not texts:
            return []

        # 硅基流动限制单次 max=64，我们设为 50 以保万无一失
        API_BATCH_LIMIT = 50
        all_embeddings = []

        try:
            # 循环切片处理：range(0, 总数, 步长)
            for i in range(0, len(texts), API_BATCH_LIMIT):
                chunk = texts[i: i + API_BATCH_LIMIT]

                # 发送分片请求
                resp = await self.emb_client.embeddings.create(
                    input=chunk,
                    model=self.emb_model
                )

                # 收集结果
                # resp.data 是按顺序返回的，直接 extend 即可
                chunk_embeddings = [data.embedding for data in resp.data]
                all_embeddings.extend(chunk_embeddings)

            return all_embeddings

        except Exception as e:
            logger.error(f"Batch Embedding API Error: {e}")
            # 如果中间失败了，返回空列表，触发上层重试机制
            return []

    async def batch_insert(self, texts: list[str], session_id: str):
        """
        批量插入聊天记录 (用于 utils.py 中的历史数据向量化)
        """
        await self._ensure_collections()
        if not texts:
            return

        # 1. 批量获取向量
        try:
            vectors = await self._get_batch_text_embeddings(texts)
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            return

        if len(vectors) != len(texts):
            logger.error(f"向量数量({len(vectors)})与文本数量({len(texts)})不匹配，跳过本批次")
            return

        # 2. 构造 Qdrant Points
        points = []
        current_time = int(time.time())

        for text, vector in zip(texts, vectors):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),  # 生成 UUID
                vector=vector,
                payload={
                    "session_id": session_id,
                    "text": text,
                    "created_at": current_time
                }
            ))

        # 3. 批量写入 Qdrant
        # Qdrant 的 upsert 本身就支持批量，效率很高
        try:
            await self.client.upsert(
                collection_name=self.chat_col,
                points=points,
                wait=True  # 批量插入建议等待确认，保证数据一致性
            )
            logger.info(f"成功批量插入 {len(points)} 条记录到 Qdrant")
        except Exception as e:
            logger.error(f"Qdrant 批量写入失败: {e}")
            raise e  # 抛出异常让 utils.py 的重试机制捕获

    async def search_meme(self, description: str) -> list[int]:
        """
        根据描述搜表情包
        Text -> Clip Vector -> Search Qdrant -> Return IDs
        """
        await self._ensure_collections()
        # 1. 文本转向量
        vector = await self._get_qwen_vl_embedding(text=description)
        if not vector:
            return []

        # 2. Qdrant 搜索
        search_result = await self.client.query_points(
            collection_name=self.media_col,
            query=vector,
            limit=10,
            with_payload=False
        )
        if not search_result or not search_result.points:
            return []
        # 3. 只返回 ID 列表
        return [int(point.id) for point in search_result.points]

    async def search_similar_meme(self, file_path: str, limit: int = 6) -> list[int] | None:
        """
        根据图片找图片 (猜你喜欢/找相似)
        ID -> Retrieve Vector -> Search Qdrant -> Return IDs
        """
        await self._ensure_collections()

        target_vector = await self._get_qwen_vl_embedding(image_source=file_path)
        if not target_vector:
            return []

        search_result = await self.client.query_points(
            collection_name=self.media_col,
            query=target_vector,
            limit=limit
        )

        if not search_result or not search_result.points:
            return []

        # 返回匹配到的表情包 ID
        return [int(point.id) for point in search_result.points]


# 实例化单例
DB = VectorDBOperator()
