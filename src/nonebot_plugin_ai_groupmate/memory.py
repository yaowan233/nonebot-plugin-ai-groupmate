import os
import re
import math
import time
import uuid
import base64
import random
import asyncio
import mimetypes
from collections.abc import Sequence, Collection

import httpx
from openai import AsyncOpenAI
from nonebot.log import logger
from qdrant_client import AsyncQdrantClient, models

from .runtime_config import get_runtime_config

plugin_config = get_runtime_config()

QWEN_VL_EMBEDDING_MODEL = "qwen3-vl-embedding"
MEDIA_VECTOR_SIZE = 2560
# SQL 中的 embedding_version 同时标识向量结构和当前表情包向量化模式。
# 两种模式使用不同版本，并在模式切换时按“不等于当前版本”重新入库。
MEDIA_MULTIMODAL_EMBEDDING_VERSION = 3
MEDIA_TEXT_EMBEDDING_VERSION = 4
# 保留原常量，兼容现有的多模态向量版本引用。
MEDIA_EMBEDDING_VERSION = MEDIA_MULTIMODAL_EMBEDDING_VERSION
MEDIA_TEXT_VECTOR = "text"
MEDIA_IMAGE_VECTOR = "image"
# text 模式（meme_embedding_mode="text"）专用：纯文本向量集合
MEDIA_TEXT_COL = "media_collection_text"
# 默认维度用于兼容未设置 embedding_dimension 的旧配置和测试替身。
MEDIA_TEXT_VECTOR_SIZE = 1024
MEME_SEARCH_POOL_SIZE = 50
MEME_RRF_K = 60
MEME_LEGACY_ROUTE_WEIGHT = 0.35
MEME_CONTEXT_VISUAL_ROUTE_WEIGHT = 0.85
MEME_CONTENT_VISUAL_ROUTE_WEIGHT = 0.65
MEME_LEGACY_ROUTE_QUOTA = 3
MEME_LEGACY_ROUTE_WINDOW = 15
MEME_QDRANT_ROUTE_TIMEOUT_SECONDS = 8.0
MEME_GROUP_USAGE_WEIGHT = 0.35
MEME_GROUP_USAGE_MIN_USES = 2
MEME_SAMPLE_RANK_SCALE = 18.0
MEME_TEXT_QUERY_INSTRUCT = (
    "Retrieve meme images by all requested constraints. Preserve exact quotes, "
    "meme references, named characters or IP, visible appearance, objects, actions, "
    "scene and style, as well as emotion, reaction and conversational intent."
)
MEME_DRAGON_IMAGE_JARGON_EXPLANATION = (
    "术语释义：‘龙图’特指黑白熊猫头、熊猫脸或人脸熊猫头的网络梗表情包，"
    "常搭配文字并以‘某某龙’命名；不是动物龙、卡通龙、龙图案、龙图标或末影龙。"
)
MEME_DRAGON_IMAGE_JARGON_PATTERN = re.compile(r"龙图(?!案|标|像|腾)")


def expand_meme_search_terms(description: str) -> str:
    """把小圈子梗名展开成图库标注能识别的视觉与语义条件。"""
    if (
        MEME_DRAGON_IMAGE_JARGON_EXPLANATION in description
        or not MEME_DRAGON_IMAGE_JARGON_PATTERN.search(description)
    ):
        return description
    return (
        f"{MEME_DRAGON_IMAGE_JARGON_EXPLANATION}\n"
        f"用户原始搜索：{description}"
    )


class VectorDBOperator:
    # 类级默认值，保证绕过 _configure 的场景（如测试）也能访问
    enabled: bool = False
    text_only: bool = False
    effective_meme_embedding_mode: str = "multimodal"
    media_embedding_version: int = MEDIA_MULTIMODAL_EMBEDDING_VERSION
    media_text_col = MEDIA_TEXT_COL
    text_embedding_dimension: int = MEDIA_TEXT_VECTOR_SIZE

    def __init__(self):
        self._configure()

    def _configure(self) -> None:
        self.effective_meme_embedding_mode = (
            "text"
            if plugin_config.meme_embedding_mode == "text"
            or not plugin_config.qwen_token
            else "multimodal"
        )
        self.text_only = self.effective_meme_embedding_mode == "text"
        self.media_embedding_version = (
            MEDIA_TEXT_EMBEDDING_VERSION
            if self.text_only
            else MEDIA_MULTIMODAL_EMBEDDING_VERSION
        )
        self.enabled = bool(plugin_config.qdrant_uri)
        if not self.enabled:
            logger.info("未配置 qdrant_uri，向量库功能已禁用")
            return

        # 1. 初始化 Qdrant 客户端
        self.client = AsyncQdrantClient(
            url=plugin_config.qdrant_uri,
            api_key=plugin_config.qdrant_api_key,
            timeout=60
        )

        self.chat_col = "chat_collection"
        # v2 及更早版本的图文融合向量，迁移期间继续作为回退召回源。
        self.media_col = "media_collection"
        # v3 将描述文本和原图拆成独立向量，避免视觉信息稀释梗和台词。
        self.media_multivector_col = "media_collection_v3"
        # text 模式：纯文本向量集合（BGE-M3，1024 维）
        self.media_text_col = MEDIA_TEXT_COL
        # 多模态是否启用。text 模式不依赖 qwen_token，也不需要多模态 embedding。
        # 未配置 qwen_token 时强制降级为 text 模式，避免空 token 调用 qwen3-vl 报错。
        if self.text_only:
            if plugin_config.meme_embedding_mode != "text":
                logger.warning("qwen_token 未配置，强制启用 text 模式")
            logger.info("表情包向量化模式: text（纯文本向量，图找图不可用）")
        else:
            logger.info("表情包向量化模式: multimodal（需要配置 qwen_token）")

        # 2. Embedding API (用于文本 -> 向量)
        # 使用硅基流动/OpenAI兼容接口
        # AsyncOpenAI client 会自动在 base_url 后追加 /embeddings，
        # 兼容用户直接填完整路径 (…/v1/embeddings) 的写法，去掉尾部重复路径。
        embedding_base_url = plugin_config.embedding_base_url.rstrip("/")
        if embedding_base_url.endswith("/embeddings"):
            embedding_base_url = embedding_base_url[: -len("/embeddings")]
        self.emb_client = AsyncOpenAI(
            api_key=plugin_config.embedding_api_key,
            base_url=embedding_base_url
        )
        self.qwen_http_client = httpx.AsyncClient(timeout=60.0)
        self.emb_model = (
            str(getattr(plugin_config, "embedding_model", "BAAI/bge-m3")).strip()
            or "BAAI/bge-m3"
        )
        self.text_embedding_dimension = int(
            getattr(
                plugin_config,
                "embedding_dimension",
                MEDIA_TEXT_VECTOR_SIZE,
            )
        )

        # 3. Rerank API 配置
        self.rerank_url = plugin_config.rerank_api_url
        self.rerank_key = plugin_config.rerank_api_key

        self._init_lock = asyncio.Lock()
        self._collections_ready = False

    async def close(self) -> None:
        """关闭向量库相关连接。"""
        qdrant_client = getattr(self, "client", None)
        embedding_client = getattr(self, "emb_client", None)
        qwen_http_client = getattr(self, "qwen_http_client", None)
        if qdrant_client is not None:
            try:
                await qdrant_client.close()
            except Exception:
                logger.exception("关闭旧 Qdrant 客户端失败")
        if embedding_client is not None:
            try:
                await embedding_client.close()
            except Exception:
                logger.exception("关闭旧 Embedding 客户端失败")
        if qwen_http_client is not None:
            try:
                await qwen_http_client.aclose()
            except Exception:
                logger.exception("关闭旧 Qwen Embedding 客户端失败")

    async def reconfigure(self) -> None:
        """重建连接，使启动阶段加载的 WebUI 配置真正生效。"""
        await self.close()
        self._configure()

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
                        size=self.text_embedding_dimension,
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
            if self.text_only:
                if not await self.client.collection_exists(self.media_text_col):
                    await self.client.create_collection(
                        collection_name=self.media_text_col,
                        vectors_config=models.VectorParams(
                            size=self.text_embedding_dimension,
                            distance=models.Distance.COSINE
                        ),
                    )
                    logger.info(f"Qdrant集合 {self.media_text_col} 已创建")
            else:
                if not await self.client.collection_exists(self.media_col):
                    await self.client.create_collection(
                        collection_name=self.media_col,
                        vectors_config=models.VectorParams(
                            size=MEDIA_VECTOR_SIZE,
                            distance=models.Distance.COSINE
                        ),
                    )
                    logger.info(f"Qdrant集合 {self.media_col} 已创建")

                if not await self.client.collection_exists(self.media_multivector_col):
                    await self.client.create_collection(
                        collection_name=self.media_multivector_col,
                        vectors_config={
                            MEDIA_TEXT_VECTOR: models.VectorParams(
                                size=MEDIA_VECTOR_SIZE,
                                distance=models.Distance.COSINE,
                            ),
                            MEDIA_IMAGE_VECTOR: models.VectorParams(
                                size=MEDIA_VECTOR_SIZE,
                                distance=models.Distance.COSINE,
                            ),
                        },
                    )
                    logger.info(f"Qdrant集合 {self.media_multivector_col} 已创建")

            self._collections_ready = True

    def _validate_text_embedding_dimension(
        self, embedding: list[float]
    ) -> list[float] | None:
        if len(embedding) == self.text_embedding_dimension:
            return embedding
        logger.error(
            "Embedding API 返回维度不匹配: "
            f"model={self.emb_model}, expected={self.text_embedding_dimension}, "
            f"actual={len(embedding)}"
        )
        return None

    async def _get_text_embedding(self, text: str) -> list[float] | None:
        """调用 API 获取配置的文本 Dense 向量。"""
        try:
            resp = await self.emb_client.embeddings.create(
                input=[text],
                model=self.emb_model
            )
            return self._validate_text_embedding_dimension(resp.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding API Error: {e}")
            return None

    @staticmethod
    def _build_qwen_vl_contents(
        text: str = "",
        image_source: str = "",
    ) -> list[dict[str, str]]:
        contents: list[dict[str, str]] = []
        if text:
            contents.append({"text": text})

        if image_source:
            if os.path.isfile(image_source):
                mime_type, _ = mimetypes.guess_type(image_source)
                mime_type = mime_type or "image/jpeg"
                with open(image_source, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                    image_value = f"data:{mime_type};base64,{base64_data}"
            elif image_source.startswith("data:image"):
                image_value = image_source
            elif image_source.startswith("http://") or image_source.startswith("https://"):
                image_value = image_source
            else:
                image_value = f"data:image/jpeg;base64,{image_source}"
            contents.append({"image": image_value})
        return contents

    async def _request_qwen_vl_embeddings(
        self,
        contents: Sequence[dict[str, str]],
        *,
        enable_fusion: bool = False,
        instruct: str = "",
    ) -> list[list[float]] | None:
        if not contents:
            logger.warning("Aliyun Embedding: contents 为空")
            return None

        aliyun_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        headers = {
            "Authorization": f"Bearer {plugin_config.qwen_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": QWEN_VL_EMBEDDING_MODEL,
            "input": {"contents": list(contents)},
            "parameters": {
                "dimension": MEDIA_VECTOR_SIZE,
            },
        }
        if enable_fusion:
            payload["parameters"]["enable_fusion"] = True
        if instruct:
            payload["parameters"]["instruct"] = instruct

        max_retries = 3
        client = getattr(self, "qwen_http_client", None)
        if client is None or getattr(client, "is_closed", False):
            client = httpx.AsyncClient(timeout=60.0)
            self.qwen_http_client = client

        for attempt in range(max_retries):
            try:
                resp = await client.post(aliyun_url, json=payload, headers=headers)

                if resp.status_code != 200:
                    logger.error(f"Aliyun API Error {resp.status_code}: {resp.text}")
                    # 限流属于临时错误，按 Retry-After 或指数退避后重试。
                    if resp.status_code == 429:
                        if attempt == max_retries - 1:
                            return None
                        retry_after = resp.headers.get("Retry-After")
                        try:
                            wait_time = float(retry_after) if retry_after else 0.0
                        except (TypeError, ValueError):
                            wait_time = 0.0
                        await asyncio.sleep(max(wait_time, 2 ** attempt) + random.random())
                        continue
                    # 其余 4xx 通常是参数错误、内容违规或欠费，无需重试。
                    if 400 <= resp.status_code < 500:
                        return None
                    resp.raise_for_status()

                response_data = resp.json()
                embeddings = response_data.get("output", {}).get("embeddings", [])
                expected_count = 1 if enable_fusion else len(contents)
                if len(embeddings) != expected_count:
                    logger.error(
                        "Aliyun Embedding 返回向量数量异常: "
                        f"expected={expected_count}, actual={len(embeddings)}"
                    )
                    return None

                if enable_fusion and embeddings[0].get("type") not in {"fusion", "fused"}:
                    logger.error(
                        "Aliyun Embedding 未返回图文融合向量: "
                        f"type={embeddings[0].get('type')!r}"
                    )
                    return None

                # 独立模式按 index 对齐输入；qwen3-vl-embedding 的 type 可能统一为 vl。
                indexed_embeddings = list(enumerate(embeddings))
                indexed_embeddings.sort(
                    key=lambda item: int(item[1].get("index", item[0]))
                )
                vectors: list[list[float]] = []
                for _, embedding_data in indexed_embeddings:
                    embedding = embedding_data.get("embedding")
                    if not isinstance(embedding, list) or len(embedding) != MEDIA_VECTOR_SIZE:
                        actual_size = len(embedding) if isinstance(embedding, list) else None
                        logger.error(
                            "Aliyun Embedding 返回维度异常: "
                            f"expected={MEDIA_VECTOR_SIZE}, actual={actual_size}"
                        )
                        return None
                    vectors.append(embedding)
                return vectors

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout) as e:
                is_last_attempt = (attempt == max_retries - 1)

                if is_last_attempt:
                    logger.error(f"Aliyun API 重试3次后最终失败: {repr(e)}")
                    return None
                wait_time = 2 * (attempt + 1)  # 2秒, 4秒...
                logger.warning(f"Aliyun API 连接抖动 ({repr(e)})，正在第 {attempt + 1} 次重试...")
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Aliyun API 未知异常: {e}")
                return None

        return None

    async def _get_qwen_vl_embedding(
        self,
        text: str = "",
        image_source: str = "",
        *,
        instruct: str = "",
    ) -> list[float] | None:
        """获取单模态向量，或在同时传入图文时获取旧版融合向量。"""
        contents = self._build_qwen_vl_contents(text, image_source)
        vectors = await self._request_qwen_vl_embeddings(
            contents,
            enable_fusion=bool(text and image_source),
            instruct=instruct,
        )
        if not vectors or len(vectors) != 1:
            return None
        return vectors[0]

    async def _get_qwen_vl_independent_pair(
        self,
        text: str,
        image_source: str,
    ) -> tuple[list[float], list[float]] | None:
        """在一次请求中分别生成描述文本向量与图片向量。"""
        if not text or not image_source:
            return None
        contents = self._build_qwen_vl_contents(text, image_source)
        vectors = await self._request_qwen_vl_embeddings(contents)
        if not vectors or len(vectors) != 2:
            return None
        return vectors[0], vectors[1]


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
        if not self.enabled:
            return "未找到相关历史记录"
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

    async def insert_media(self, media_id: int, image_url: str, description: str) -> bool:
        """插入新表情包 (新图入库用)"""
        if not self.enabled:
            return False
        await self._ensure_collections()

        if self.text_only:
            # text 模式：只对描述文本做 BGE-M3 向量化，不依赖多模态 embedding
            if not description:
                logger.warning(f"text 模式缺少描述，跳过 {media_id}")
                return False
            vector = await self._get_text_embedding(description)
            if not vector:
                return False
            await self.client.upsert(
                collection_name=self.media_text_col,
                points=[
                    models.PointStruct(
                        id=media_id,
                        vector=vector,
                        payload={
                            "created_at": int(time.time()),
                            "embedding_version": self.media_embedding_version,
                        }
                    )
                ],
                wait=True,
            )
            return True

        vectors = await self._get_qwen_vl_independent_pair(description, image_url)
        if not vectors:
            return False
        text_vector, image_vector = vectors

        await self.client.upsert(
            collection_name=self.media_multivector_col,
            points=[
                models.PointStruct(
                    id=media_id,  # 保持 Int ID
                    vector={
                        MEDIA_TEXT_VECTOR: text_vector,
                        MEDIA_IMAGE_VECTOR: image_vector,
                    },
                    payload={
                        "created_at": int(time.time()),
                        "embedding_version": self.media_embedding_version,
                    }
                )
            ],
            wait=True,
        )
        return True

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
                chunk_embeddings = []
                for data in resp.data:
                    embedding = self._validate_text_embedding_dimension(
                        data.embedding
                    )
                    if embedding is None:
                        return []
                    chunk_embeddings.append(embedding)
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
        if not self.enabled:
            return
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

    @staticmethod
    def _weighted_sample_meme_ids(
        candidates: Sequence[tuple[int, float]],
        limit: int,
    ) -> list[int]:
        """按融合排名衰减做无放回抽样，避免相同查询永远返回同一顺序。"""
        remaining = list(candidates)
        selected: list[int] = []
        while remaining and len(selected) < limit:
            weights = [
                math.exp(-rank / MEME_SAMPLE_RANK_SCALE)
                for rank in range(len(remaining))
            ]
            index = random.choices(range(len(remaining)), weights=weights, k=1)[0]
            media_id, _ = remaining.pop(index)
            selected.append(media_id)
        return selected

    @classmethod
    def _diversify_meme_results(
        cls,
        points: Sequence[models.ScoredPoint],
        *,
        exclude_ids: Collection[int],
        limit: int,
    ) -> list[int]:
        candidates = [
            (int(point.id), float(getattr(point, "score", 0.0)))
            for point in points
        ]
        return cls._diversify_meme_candidates(
            candidates,
            exclude_ids=exclude_ids,
            limit=limit,
        )

    @staticmethod
    def _merge_weighted_meme_search_routes(
        routes: Sequence[tuple[Sequence[models.ScoredPoint], float]],
    ) -> list[tuple[int, float]]:
        """使用加权 RRF 合并任意数量的文本、视觉与兼容召回路线。"""
        merged: dict[int, float] = {}
        for points, weight in routes:
            for rank, point in enumerate(points, start=1):
                media_id = int(point.id)
                merged[media_id] = merged.get(media_id, 0.0) + (
                    weight / (MEME_RRF_K + rank)
                )
        return sorted(merged.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _merge_meme_search_routes(
        primary_points: Sequence[models.ScoredPoint],
        legacy_points: Sequence[models.ScoredPoint],
    ) -> list[tuple[int, float]]:
        """使用加权 RRF 合并独立向量与旧融合向量召回。"""
        return VectorDBOperator._merge_weighted_meme_search_routes([
            (primary_points, 1.0),
            (legacy_points, MEME_LEGACY_ROUTE_WEIGHT),
        ])

    @staticmethod
    def _reserve_meme_route_candidates(
        candidates: Sequence[tuple[int, float]],
        route_points: Sequence[models.ScoredPoint],
        *,
        exclude_ids: Collection[int],
        quota: int,
        window: int,
    ) -> list[tuple[int, float]]:
        """在靠前窗口为互补路线保留少量独立候选。"""
        ranked = list(candidates)
        if not ranked or not route_points or quota <= 0 or window <= 0:
            return ranked

        excluded = set(exclude_ids)
        route_only_ids = [
            int(point.id)
            for point in route_points
            if int(point.id) not in excluded
        ]
        if not route_only_ids:
            return ranked

        window_size = min(window, len(ranked))
        head = ranked[:window_size]
        route_only_set = set(route_only_ids)
        existing_ids = {
            media_id for media_id, _ in head if media_id in route_only_set
        }
        needed = max(0, quota - len(existing_ids))
        if needed == 0:
            return ranked

        score_by_id = dict(ranked)
        promoted_ids: list[int] = []
        for media_id in route_only_ids:
            if (
                media_id in score_by_id
                and media_id not in existing_ids
                and media_id not in promoted_ids
            ):
                promoted_ids.append(media_id)
                if len(promoted_ids) >= needed:
                    break
        if not promoted_ids:
            return ranked

        promoted_set = set(promoted_ids)
        retained_head = [item for item in head if item[0] not in promoted_set]
        retained_head = retained_head[:window_size - len(promoted_ids)]
        new_head = retained_head + [
            (media_id, score_by_id[media_id]) for media_id in promoted_ids
        ]
        new_head_ids = {media_id for media_id, _ in new_head}
        tail = [item for item in ranked if item[0] not in new_head_ids]
        return new_head + tail

    @staticmethod
    def apply_group_usage_boost(
        candidates: Sequence[tuple[int, float]],
        usage_counts: dict[int, int],
    ) -> list[tuple[int, float]]:
        """把群内人类用图频率作为一条低权重 RRF 路线参与候选重排。"""
        if not candidates or not usage_counts:
            return list(candidates)

        scores = {
            media_id: 1.0 / (MEME_RRF_K + rank)
            for rank, (media_id, _) in enumerate(candidates, start=1)
        }
        semantic_rank = {
            media_id: rank
            for rank, (media_id, _) in enumerate(candidates, start=1)
        }
        popular_ids = sorted(
            (
                media_id
                for media_id, _ in candidates
                if usage_counts.get(media_id, 0) >= MEME_GROUP_USAGE_MIN_USES
            ),
            key=lambda media_id: (
                -usage_counts[media_id],
                semantic_rank[media_id],
            ),
        )
        for rank, media_id in enumerate(popular_ids, start=1):
            scores[media_id] += MEME_GROUP_USAGE_WEIGHT / (MEME_RRF_K + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    async def _search_media_routes(
        self,
        vector: list[float],
        *,
        vector_name: str,
        limit: int,
        visual_route_weight: float = MEME_CONTEXT_VISUAL_ROUTE_WEIGHT,
    ) -> list[tuple[int, float]]:
        query_specs: list[tuple[str, str | None, float, str]] = [
            (self.media_multivector_col, vector_name, 1.0, "独立媒体向量"),
        ]
        if vector_name == MEDIA_TEXT_VECTOR:
            # Qwen VL 的文本和图片向量位于同一语义空间。文字搜图时同时查询
            # 原图视觉向量，才能召回描述中漏写的角色、外观、物体与构图。
            query_specs.append((
                self.media_multivector_col,
                MEDIA_IMAGE_VECTOR,
                visual_route_weight,
                "跨模态视觉向量",
            ))
        query_specs.append((
            self.media_col,
            None,
            MEME_LEGACY_ROUTE_WEIGHT,
            "旧媒体融合向量",
        ))

        calls = []
        for collection_name, using, _, _ in query_specs:
            kwargs = {
                "collection_name": collection_name,
                "query": vector,
                "limit": limit,
                "with_payload": False,
                "timeout": math.ceil(MEME_QDRANT_ROUTE_TIMEOUT_SECONDS),
            }
            if using is not None:
                kwargs["using"] = using
            calls.append(asyncio.wait_for(
                self.client.query_points(**kwargs),
                timeout=MEME_QDRANT_ROUTE_TIMEOUT_SECONDS + 1.0,
            ))
        results = await asyncio.gather(*calls, return_exceptions=True)

        routes: list[tuple[Sequence[models.ScoredPoint], float]] = []
        primary_ids: set[int] = set()
        legacy_points: Sequence[models.ScoredPoint] = []
        for (collection_name, _, weight, label), result in zip(query_specs, results):
            if isinstance(result, BaseException):
                error_detail = (
                    f"超过 {MEME_QDRANT_ROUTE_TIMEOUT_SECONDS:.1f}s"
                    if isinstance(result, TimeoutError)
                    else str(result).strip() or repr(result)
                )
                logger.warning(
                    f"{label}检索失败，跳过该路线: {error_detail}"
                )
                continue
            points = result.points if result else []
            routes.append((points, weight))
            if collection_name == self.media_col:
                legacy_points = points
            else:
                primary_ids.update(int(point.id) for point in points)

        merged = self._merge_weighted_meme_search_routes(routes)
        return self._reserve_meme_route_candidates(
            merged,
            legacy_points,
            exclude_ids=primary_ids,
            quota=MEME_LEGACY_ROUTE_QUOTA,
            window=MEME_LEGACY_ROUTE_WINDOW,
        )

    @classmethod
    def _diversify_meme_candidates(
        cls,
        candidates: Sequence[tuple[int, float]],
        *,
        exclude_ids: Collection[int],
        limit: int,
    ) -> list[int]:
        fresh = [item for item in candidates if item[0] not in exclude_ids]
        recent = [item for item in candidates if item[0] in exclude_ids]
        selected = cls._weighted_sample_meme_ids(fresh, limit)
        if len(selected) < limit:
            selected.extend(
                cls._weighted_sample_meme_ids(recent, limit - len(selected))
            )
        return selected

    async def search_meme(
        self,
        description: str,
        *,
        limit: int = 5,
        exclude_ids: Collection[int] = (),
    ) -> list[int]:
        """
        根据描述搜表情包
        Text -> Qwen Vector -> Search Qdrant -> Return IDs
        """
        if not self.enabled:
            return []
        query_limit = min(
            100,
            max(MEME_SEARCH_POOL_SIZE, limit * 8, limit + len(exclude_ids)),
        )
        candidates = await self.search_meme_candidates(
            description,
            limit=query_limit,
        )
        if not candidates:
            return []
        return self._diversify_meme_candidates(
            candidates,
            exclude_ids=exclude_ids,
            limit=limit,
        )

    async def search_meme_candidates(
        self,
        description: str,
        *,
        limit: int = MEME_SEARCH_POOL_SIZE,
        strict_content_match: bool = False,
    ) -> list[tuple[int, float]]:
        """返回多路融合后的候选，供上层结合群热度再次排序。"""
        if not self.enabled:
            return []
        await self._ensure_collections()
        expanded_description = expand_meme_search_terms(description)

        if self.text_only:
            # text 模式：BGE-M3 文本向量 -> 单路线检索 media_collection_text
            vector = await self._get_text_embedding(expanded_description)
            if not vector:
                return []
            search_result = await self.client.query_points(
                collection_name=self.media_text_col,
                query=vector,
                limit=limit,
                with_payload=False,
                timeout=math.ceil(MEME_QDRANT_ROUTE_TIMEOUT_SECONDS),
            )
            if not search_result or not search_result.points:
                return []
            return [
                (int(point.id), float(getattr(point, "score", 0.0)))
                for point in search_result.points
            ]

        vector = await self._get_qwen_vl_embedding(
            text=expanded_description,
            instruct=MEME_TEXT_QUERY_INSTRUCT,
        )
        if not vector:
            return []

        return await self._search_media_routes(
            vector,
            vector_name=MEDIA_TEXT_VECTOR,
            limit=limit,
            visual_route_weight=(
                MEME_CONTENT_VISUAL_ROUTE_WEIGHT
                if strict_content_match
                else MEME_CONTEXT_VISUAL_ROUTE_WEIGHT
            ),
        )

    async def search_similar_meme(
        self,
        file_path: str,
        limit: int = 6,
        *,
        exclude_ids: Collection[int] = (),
    ) -> list[int] | None:
        """
        根据图片找图片 (猜你喜欢/找相似)
        ID -> Retrieve Vector -> Search Qdrant -> Return IDs
        """
        if not self.enabled:
            return []
        if self.text_only:
            logger.warning("text 模式下图找图 (search_similar_meme) 不可用")
            return []
        await self._ensure_collections()

        target_vector = await self._get_qwen_vl_embedding(image_source=file_path)
        if not target_vector:
            return []

        query_limit = min(
            100,
            max(MEME_SEARCH_POOL_SIZE, limit * 8, limit + len(exclude_ids)),
        )
        candidates = await self._search_media_routes(
            target_vector,
            vector_name=MEDIA_IMAGE_VECTOR,
            limit=query_limit,
        )
        if not candidates:
            return []

        return self._diversify_meme_candidates(
            candidates,
            exclude_ids=exclude_ids,
            limit=limit,
        )


# 实例化单例
DB = VectorDBOperator()
