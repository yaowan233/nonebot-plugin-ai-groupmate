import os
import re
import math
import time
import uuid
import base64
import random
import asyncio
import hashlib
import mimetypes
import unicodedata
from typing import Any, NoReturn
from datetime import datetime, timedelta
from collections import Counter
from collections.abc import Mapping, Sequence, Collection

import httpx
from openai import AsyncOpenAI, RateLimitError, BadRequestError
from nonebot.log import logger
from qdrant_client import AsyncQdrantClient, models

from .runtime_config import get_runtime_config

plugin_config = get_runtime_config()

QWEN_VL_EMBEDDING_MODEL = "qwen3-vl-embedding"
MEDIA_VECTOR_SIZE = 2560
LEGACY_TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
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
# Qdrant collection metadata keys used to pin an embedding space to a collection.
EMBEDDING_MODEL_METADATA_KEY = "embedding_model"
EMBEDDING_DIMENSION_METADATA_KEY = "embedding_dimension"
# 默认维度用于兼容未设置 embedding_dimension 的旧配置和测试替身。
MEDIA_TEXT_VECTOR_SIZE = 1024
CHAT_COLLECTION = "chat_collection_v2"
CHAT_DENSE_VECTOR = "dense"
CHAT_SPARSE_VECTOR = "lexical"
CHAT_INDEX_VERSION = 2
CHAT_SEARCH_CANDIDATE_LIMIT = 40
CHAT_RERANK_POOL_SIZE = 12
CHAT_SEARCH_RESULT_LIMIT = 5
CHAT_RESULT_FRAGMENT_MAX_CHARS = 800
CHAT_RESULT_TOTAL_MAX_CHARS = 3200
CHAT_RRF_K = 60
CHAT_LEXICAL_ROUTE_WEIGHT = 0.9
CHAT_RECENCY_ROUTE_WEIGHT = 0.35
CHAT_RECENCY_QUERY_PATTERN = re.compile(
    r"(?:刚才|最近|近期|今天|昨天|前天|本周|上周|这个月|本月|上个月|最新|上次)"
)
CHAT_TIMESTAMP_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
CHAT_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
MEME_SEARCH_POOL_SIZE = 50
MEME_RRF_K = 60
MEME_CONTEXT_VISUAL_ROUTE_WEIGHT = 0.85
MEME_CONTENT_VISUAL_ROUTE_WEIGHT = 0.65
MEME_QDRANT_ROUTE_TIMEOUT_SECONDS = 20.0
MEME_QDRANT_HNSW_EF = 16
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

# 聊天记忆回填会连续发送大批量 Embedding 请求。每个上下文最多
# 约 450 tokens，因此 20 条 / 1.5 秒的最坏速率约为 36 万 TPM，为
# SiliconFlow 公开的最低 50 万 Embedding TPM 留出实时检索余量。
TEXT_EMBEDDING_BATCH_SIZE = 20
TEXT_EMBEDDING_BATCH_MIN_INTERVAL_SECONDS = 1.5
TEXT_EMBEDDING_RATE_LIMIT_MAX_RETRIES = 4
TEXT_EMBEDDING_RATE_LIMIT_BASE_DELAY_SECONDS = 15.0
TEXT_EMBEDDING_RATE_LIMIT_MAX_DELAY_SECONDS = 60.0


class CollectionEmbeddingConfigMismatchError(RuntimeError):
    """Raised when a collection cannot be used with the active embedding space."""


class EmbeddingProviderUnavailableError(RuntimeError):
    """Raised when the embedding endpoint cannot be checked or reached."""


class CollectionMetadataBackfillError(RuntimeError):
    """Raised when a legacy collection cannot be marked with its embedding space."""


def expand_chat_search_query(query: str, *, now: datetime | None = None) -> str:
    """把常见相对时间词展开成聊天记录中实际存在的日期字符串。"""
    normalized_query = query.strip()
    if not normalized_query:
        return normalized_query

    current = now or datetime.now()
    current_date = current.date()
    ranges: list[str] = []

    for term, days_ago in (("前天", 2), ("昨天", 1), ("今天", 0)):
        if term in normalized_query:
            target = current_date - timedelta(days=days_ago)
            ranges.append(f"{term}={target.isoformat()}")

    if "本周" in normalized_query:
        start = current_date - timedelta(days=current_date.weekday())
        ranges.append(f"本周={start.isoformat()}至{current_date.isoformat()}")
    if "上周" in normalized_query:
        this_week = current_date - timedelta(days=current_date.weekday())
        start = this_week - timedelta(days=7)
        end = this_week - timedelta(days=1)
        ranges.append(f"上周={start.isoformat()}至{end.isoformat()}")

    month_start = current_date.replace(day=1)
    if "这个月" in normalized_query or "本月" in normalized_query:
        ranges.append(f"本月={month_start.isoformat()}至{current_date.isoformat()}")
    if "上个月" in normalized_query:
        previous_month_end = month_start - timedelta(days=1)
        previous_month_start = previous_month_end.replace(day=1)
        ranges.append(
            f"上个月={previous_month_start.isoformat()}至"
            f"{previous_month_end.isoformat()}"
        )

    if not ranges:
        return normalized_query
    return f"{normalized_query}\n检索时间范围：{'；'.join(ranges)}"


def parse_chat_search_time_range(
    query: str,
    *,
    now: datetime | None = None,
) -> tuple[int, int] | None:
    """把明确时间条件转换为本地时区 Unix 时间范围。"""
    normalized_query = query.strip()
    if not normalized_query:
        return None

    current = now or datetime.now()
    current_date = current.date()
    ranges: list[tuple[datetime, datetime]] = []

    dates = CHAT_DATE_PATTERN.findall(normalized_query)
    if dates:
        try:
            parsed_dates = [datetime.strptime(value, "%Y-%m-%d") for value in dates[:2]]
        except ValueError:
            parsed_dates = []
        if parsed_dates:
            start = parsed_dates[0]
            end_date = parsed_dates[-1]
            ranges.append((
                start,
                end_date + timedelta(days=1) - timedelta(seconds=1),
            ))
    for term, days_ago in (("前天", 2), ("昨天", 1), ("今天", 0)):
        if term not in normalized_query:
            continue
        start = datetime.combine(
            current_date - timedelta(days=days_ago),
            datetime.min.time(),
        )
        end = current if days_ago == 0 else start + timedelta(days=1) - timedelta(seconds=1)
        ranges.append((start, end))
    if "上周" in normalized_query:
        this_week = current_date - timedelta(days=current_date.weekday())
        start = datetime.combine(this_week - timedelta(days=7), datetime.min.time())
        end = datetime.combine(this_week, datetime.min.time()) - timedelta(seconds=1)
        ranges.append((start, end))
    if "本周" in normalized_query:
        week_start = current_date - timedelta(days=current_date.weekday())
        start = datetime.combine(week_start, datetime.min.time())
        ranges.append((start, current))
    if "上个月" in normalized_query:
        month_start = current_date.replace(day=1)
        previous_month_end = month_start - timedelta(days=1)
        previous_month_start = previous_month_end.replace(day=1)
        start = datetime.combine(previous_month_start, datetime.min.time())
        end = datetime.combine(month_start, datetime.min.time()) - timedelta(seconds=1)
        ranges.append((start, end))
    if "这个月" in normalized_query or "本月" in normalized_query:
        start = datetime.combine(current_date.replace(day=1), datetime.min.time())
        ranges.append((start, current))

    if not ranges:
        return None
    start = min(value[0] for value in ranges)
    end = max(value[1] for value in ranges)
    return int(start.timestamp()), int(end.timestamp())


def _normalize_chat_search_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _chat_search_terms(text: str) -> set[str]:
    """生成适合中英文混合群聊的轻量关键词集合。"""
    normalized = _normalize_chat_search_text(text)
    terms = set(re.findall(r"[a-z0-9_]+", normalized))
    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        if len(sequence) == 1:
            terms.add(sequence)
            continue
        if len(sequence) <= 8:
            terms.add(sequence)
        terms.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return terms


def _chat_sparse_vector(text: str) -> models.SparseVector:
    """用稳定哈希生成轻量稀疏词项向量，支持中英文精确召回。"""
    term_counts = Counter(_chat_search_terms(text))
    values_by_index: dict[int, float] = {}
    for term, count in term_counts.items():
        index = int.from_bytes(
            hashlib.blake2s(term.encode("utf-8"), digest_size=4).digest(),
            "big",
        )
        values_by_index[index] = values_by_index.get(index, 0.0) + (
            1.0 + math.log(max(count, 1))
        )
    indexes = sorted(values_by_index)
    return models.SparseVector(
        indices=indexes,
        values=[values_by_index[index] for index in indexes],
    )


def _chat_context_created_at(text: str, fallback: int) -> int:
    """从上下文最后一条消息读取真实时间，旧格式则使用入库时间。"""
    matches = CHAT_TIMESTAMP_PATTERN.findall(text)
    if not matches:
        return fallback
    try:
        return int(datetime.strptime(matches[-1], "%Y-%m-%d %H:%M:%S").timestamp())
    except ValueError:
        return fallback


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
    chat_col = CHAT_COLLECTION
    media_text_col = MEDIA_TEXT_COL
    # 文本向量维度：由探测确定（配置了 embedding_dimension 则为其值，
    # 否则为模型默认输出维度），用于创建集合与校验。
    text_embedding_dimension: int = MEDIA_TEXT_VECTOR_SIZE
    # 用户配置的 embedding_dimension；None 表示不携带 dimensions 参数。
    configured_embedding_dimension: int | None = None
    emb_model: str = LEGACY_TEXT_EMBEDDING_MODEL
    _collection_validation_errors: dict[
        str, CollectionEmbeddingConfigMismatchError
    ] | None = None
    _text_embedding_validation_error: CollectionEmbeddingConfigMismatchError | None = None
    _text_embedding_probe_done: bool = False

    def __init__(self):
        self._configure()

    def _configure(self) -> None:
        self._init_lock = asyncio.Lock()
        self._embedding_batch_lock = asyncio.Lock()
        self._embedding_last_batch_request_at = 0.0
        self._ready_collections: set[str] = set()
        self._collection_validation_errors = {}
        self._text_embedding_validation_error = None
        self._text_embedding_probe_done = False
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

        self.chat_col = CHAT_COLLECTION
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
            str(
                getattr(
                    plugin_config,
                    "embedding_model",
                    LEGACY_TEXT_EMBEDDING_MODEL,
                )
            ).strip()
            or LEGACY_TEXT_EMBEDDING_MODEL
        )
        configured_dimension = getattr(
            plugin_config,
            "embedding_dimension",
            None,
        )
        self.configured_embedding_dimension = (
            int(configured_dimension) if configured_dimension else None
        )
        # 未探测前先用配置值（若配置了）；探测后会覆盖为实际维度。
        self.text_embedding_dimension = (
            self.configured_embedding_dimension
            or MEDIA_TEXT_VECTOR_SIZE
        )

        # 3. Rerank API 配置
        self.rerank_url = plugin_config.rerank_api_url
        self.rerank_key = plugin_config.rerank_api_key


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
    @staticmethod
    def _embedding_metadata(model: str, dimension: int) -> dict[str, str | int]:
        return {
            EMBEDDING_MODEL_METADATA_KEY: model,
            EMBEDDING_DIMENSION_METADATA_KEY: dimension,
        }

    @staticmethod
    def _vector_size(vector_config: Any) -> int | None:
        size = (
            vector_config.get("size")
            if isinstance(vector_config, dict)
            else getattr(vector_config, "size", None)
        )
        try:
            return int(size) if size is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _collection_vector_schema(cls, collection_info: Any) -> dict[str, int | None]:
        params = getattr(getattr(collection_info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        if isinstance(vectors, dict):
            return {
                str(vector_name): cls._vector_size(vector_config)
                for vector_name, vector_config in vectors.items()
            }
        return {"": cls._vector_size(vectors)}

    def _active_collection_specs(
        self,
    ) -> list[tuple[str, str, int, dict[str, int]]]:
        chat_schema = (
            {CHAT_DENSE_VECTOR: self.text_embedding_dimension}
            if self.chat_col == CHAT_COLLECTION
            else {"": self.text_embedding_dimension}
        )
        specs = [
            (
                self.chat_col,
                self.emb_model,
                self.text_embedding_dimension,
                chat_schema,
            )
        ]
        if self.text_only:
            specs.append((
                self.media_text_col,
                self.emb_model,
                self.text_embedding_dimension,
                {"": self.text_embedding_dimension},
            ))
        else:
            specs.append((
                self.media_multivector_col,
                QWEN_VL_EMBEDDING_MODEL,
                MEDIA_VECTOR_SIZE,
                {
                    MEDIA_TEXT_VECTOR: MEDIA_VECTOR_SIZE,
                    MEDIA_IMAGE_VECTOR: MEDIA_VECTOR_SIZE,
                },
            ))
        return specs

    def _reject_collection(
        self,
        collection_name: str,
        current_config: dict[str, Any],
        collection_config: dict[str, Any],
        reason: str,
    ) -> None:
        message = (
            "Qdrant 集合 Embedding 配置不匹配，拒绝使用: "
            f"collection={collection_name!r}, reason={reason}, "
            f"当前配置={current_config!r}, collection配置={collection_config!r}"
        )
        errors = self._get_collection_validation_errors()
        error = errors.get(collection_name)
        if error is None:
            error = CollectionEmbeddingConfigMismatchError(message)
            errors[collection_name] = error
            logger.error(message)
        raise error

    def _reject_validation(
        self,
        message: str,
    ) -> NoReturn:
        existing_error = getattr(self, "_text_embedding_validation_error", None)
        if existing_error is not None:
            raise existing_error
        error = CollectionEmbeddingConfigMismatchError(message)
        self._text_embedding_validation_error = error
        logger.error(message)
        raise error

    def _raise_if_validation_rejected(self) -> None:
        validation_error = getattr(self, "_text_embedding_validation_error", None)
        if validation_error is not None:
            raise validation_error

    def _text_collection_names(self) -> set[str]:
        names = {self.chat_col}
        if self.text_only:
            names.add(self.media_text_col)
        return names

    def _primary_media_collection_names(self) -> set[str]:
        if self.text_only:
            return {self.media_text_col}
        return {self.media_multivector_col}

    def _get_ready_collections(self) -> set[str]:
        ready_collections = getattr(self, "_ready_collections", None)
        if ready_collections is None:
            ready_collections = set()
            self._ready_collections = ready_collections
        return ready_collections

    def _get_collection_validation_errors(
        self,
    ) -> dict[str, CollectionEmbeddingConfigMismatchError]:
        errors = getattr(self, "_collection_validation_errors", None)
        if errors is None:
            errors = {}
            self._collection_validation_errors = errors
        return errors

    async def _ensure_chat_payload_indexes(self) -> None:
        for field_name, field_schema in (
            ("session_id", models.PayloadSchemaType.KEYWORD),
            ("start_at", models.PayloadSchemaType.INTEGER),
            ("end_at", models.PayloadSchemaType.INTEGER),
        ):
            await self.client.create_payload_index(
                collection_name=self.chat_col,
                field_name=field_name,
                field_schema=field_schema,
            )

    def _raise_if_collections_rejected(
        self,
        collection_names: Collection[str],
    ) -> None:
        errors = self._get_collection_validation_errors()
        for collection_name in collection_names:
            error = errors.get(collection_name)
            if error is not None:
                raise error

    async def _validate_collection(
        self,
        collection_name: str,
        expected_model: str,
        expected_dimension: int,
        expected_schema: dict[str, int],
    ) -> None:
        info = await self.client.get_collection(collection_name)
        metadata = getattr(getattr(info, "config", None), "metadata", None)
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        vector_schema = self._collection_vector_schema(info)
        params = getattr(getattr(info, "config", None), "params", None)
        sparse_vectors = getattr(params, "sparse_vectors", None)
        sparse_vector_names = (
            {str(name) for name in sparse_vectors}
            if isinstance(sparse_vectors, dict)
            else set()
        )
        current_config = {
            **self._embedding_metadata(expected_model, expected_dimension),
            "vector_schema": expected_schema,
        }
        collection_config = {
            "metadata": metadata,
            "vector_schema": vector_schema,
        }
        if collection_name == CHAT_COLLECTION:
            current_config["sparse_vectors"] = {CHAT_SPARSE_VECTOR}
            collection_config["sparse_vectors"] = sparse_vector_names

        embedding_metadata_keys = {
            EMBEDDING_MODEL_METADATA_KEY,
            EMBEDDING_DIMENSION_METADATA_KEY,
        }
        present_embedding_metadata_keys = embedding_metadata_keys & metadata.keys()
        if not present_embedding_metadata_keys:
            if vector_schema != expected_schema:
                self._reject_collection(
                    collection_name,
                    current_config,
                    collection_config,
                    "旧集合向量维度或名称不匹配，未写入新 metadata",
                )
            if (
                collection_name in self._text_collection_names()
                and expected_model != LEGACY_TEXT_EMBEDDING_MODEL
            ):
                self._reject_collection(
                    collection_name,
                    current_config,
                    collection_config,
                    "旧文本集合缺少 Embedding metadata，无法确认其模型；"
                    "当前模型不是历史默认模型，未写入新 metadata",
                )
            try:
                result = await self.client.update_collection(
                    collection_name=collection_name,
                    metadata=self._embedding_metadata(
                        expected_model,
                        expected_dimension,
                    ),
                )
                if result is False:
                    raise RuntimeError("Qdrant update_collection 返回 False")
            except Exception as exc:
                message = (
                    "旧 Qdrant 集合 metadata 补写暂时失败，本次向量操作已中止，"
                    "将在后续操作时重试: "
                    f"collection={collection_name!r}, error={exc}"
                )
                logger.warning(message)
                raise CollectionMetadataBackfillError(message) from exc
            logger.info(
                "已为旧 Qdrant 集合补充 Embedding metadata: "
                f"collection={collection_name!r}, metadata="
                f"{self._embedding_metadata(expected_model, expected_dimension)!r}"
            )
            metadata.update(self._embedding_metadata(expected_model, expected_dimension))
            collection_config["metadata"] = metadata
        elif present_embedding_metadata_keys != embedding_metadata_keys:
            self._reject_collection(
                collection_name,
                current_config,
                collection_config,
                "collection metadata 不完整",
            )

        actual_model = metadata.get(EMBEDDING_MODEL_METADATA_KEY)
        raw_dimension = metadata.get(EMBEDDING_DIMENSION_METADATA_KEY)
        if isinstance(raw_dimension, (int, str)):
            try:
                actual_dimension = int(raw_dimension)
            except ValueError:
                actual_dimension = None
        else:
            actual_dimension = None
        if (
            actual_model != expected_model
            or actual_dimension != expected_dimension
            or vector_schema != expected_schema
            or (
                collection_name == CHAT_COLLECTION
                and CHAT_SPARSE_VECTOR not in sparse_vector_names
            )
        ):
            self._reject_collection(
                collection_name,
                current_config,
                collection_config,
                "metadata 或向量 schema 不匹配",
            )

    async def _ensure_collections(
        self,
        collection_names: Collection[str] | None = None,
        *,
        validate_text_embedding: bool = True,
    ) -> None:
        """
        初始化集合：如果集合不存在，则创建并开启 Int8 量化。
        """
        collection_specs = self._active_collection_specs()
        requested_names = (
            {name for name, _, _, _ in collection_specs}
            if collection_names is None
            else set(collection_names)
        )
        self._raise_if_collections_rejected(requested_names)
        needs_text_embedding = bool(
            requested_names & self._text_collection_names()
        )
        if validate_text_embedding and needs_text_embedding:
            text_validation_error = getattr(
                self, "_text_embedding_validation_error", None
            )
            if text_validation_error is not None:
                raise text_validation_error
        ready_collections = self._get_ready_collections()
        if requested_names <= ready_collections:
            return

        async with self._init_lock:
            self._raise_if_collections_rejected(requested_names)
            if validate_text_embedding and needs_text_embedding:
                text_validation_error = getattr(
                    self, "_text_embedding_validation_error", None
                )
                if text_validation_error is not None:
                    raise text_validation_error
            if requested_names <= ready_collections:
                return

            # 探测文本向量维度（仅在涉及文本集合时）。失败时传播不可用
            # 状态，让调用方在访问文本集合前停止（而非静默跳过创建后对
            # 不存在的集合发 Qdrant 请求，产生硬错误）。多模态集合不受
            # 影响：仅当请求本身涉及文本集合时才需要探测。
            if (
                validate_text_embedding
                and needs_text_embedding
                and getattr(self, "emb_client", None) is not None
            ):
                await self._probe_text_embedding_dimension()

            # probe 可能更新了 text_embedding_dimension（未配置维度时取
            # 模型默认输出维度），重建 specs 以使用探测后的实际维度；
            # 否则会按旧的 1024 默认值创建/校验集合，schema 与后续
            # embedding upsert 不匹配。
            collection_specs = self._active_collection_specs()

            for collection_name, model, dimension, expected_schema in collection_specs:
                if collection_name not in requested_names:
                    continue
                if collection_name in ready_collections:
                    continue
                if await self.client.collection_exists(collection_name):
                    await self._validate_collection(
                        collection_name,
                        model,
                        dimension,
                        expected_schema,
                    )
                    if collection_name == CHAT_COLLECTION:
                        await self._ensure_chat_payload_indexes()
                    ready_collections.add(collection_name)
                    continue

                if expected_schema.keys() == {""}:
                    vectors_config: Any = models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                    )
                else:
                    vectors_config = {
                        vector_name: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                        for vector_name, vector_size in expected_schema.items()
                    }
                create_kwargs: dict[str, Any] = {
                    "collection_name": collection_name,
                    "vectors_config": vectors_config,
                    "metadata": self._embedding_metadata(model, dimension),
                }
                if collection_name == CHAT_COLLECTION:
                    create_kwargs["sparse_vectors_config"] = {
                        CHAT_SPARSE_VECTOR: models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    }
                await self.client.create_collection(**create_kwargs)
                if collection_name == self.chat_col:
                    if collection_name == CHAT_COLLECTION:
                        await self._ensure_chat_payload_indexes()
                    else:
                        await self.client.create_payload_index(
                            collection_name=self.chat_col,
                            field_name="session_id",
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )
                logger.info(
                    f"Qdrant集合 {collection_name} 已创建，Embedding metadata="
                    f"{self._embedding_metadata(model, dimension)!r}"
                )
                ready_collections.add(collection_name)

    async def check_collections(self) -> None:
        """在启动阶段或首次使用时校验 Qdrant 集合的 Embedding 配置。"""
        if self.enabled:
            if self.text_only:
                await self._ensure_collections()
                return
            await self._ensure_collections(
                {self.chat_col, self.media_multivector_col},
            )

    async def _probe_text_embedding_dimension(self) -> None:
        """探测文本向量维度，仅在创建文本集合前调用一次。

        配置了 embedding_dimension 时：携带 dimensions 请求，返回维度必须
        与配置一致，否则视为配置错误（模型不支持 dimensions 或配置有误）。
        未配置时：不携带 dimensions，使用模型默认输出维度建集合。
        """
        if getattr(self, "_text_embedding_probe_done", False):
            return
        try:
            if self.configured_embedding_dimension is not None:
                response = await self.emb_client.embeddings.create(
                    input=["embedding dimension validation probe"],
                    model=self.emb_model,
                    dimensions=self.configured_embedding_dimension,
                )
                if not response.data:
                    raise ValueError("Embedding API 返回空结果")
                self._validate_text_embedding_dimension(response.data[0].embedding)
            else:
                response = await self.emb_client.embeddings.create(
                    input=["embedding dimension validation probe"],
                    model=self.emb_model,
                )
                if not response.data:
                    raise ValueError("Embedding API 返回空结果")
                self.text_embedding_dimension = len(response.data[0].embedding)
            self._text_embedding_probe_done = True
        except CollectionEmbeddingConfigMismatchError:
            raise
        except BadRequestError as exc:
            # 400：请求参数不被接受。配置了维度时通常是 dimensions 不受该
            # 模型支持；未配置维度时请求不带 dimensions，更可能是模型名
            # 或接口配置错误，按场景给出不同提示。
            if self.configured_embedding_dimension is not None:
                message = (
                    "Embedding 请求被拒绝（400）：该模型可能不支持 dimensions 参数，"
                    "请移除 ai_groupmate__embedding_dimension 配置或更换模型。"
                )
            else:
                message = (
                    "Embedding 请求被拒绝（400）：请检查 ai_groupmate__embedding_model "
                    "模型名与 ai_groupmate__embedding_base_url 接口配置是否正确。"
                )
            self._reject_validation(f"{message}详情: {exc}")
        except Exception as exc:
            logger.error(
                "Embedding API 暂时不可用，跳过本次文本向量维度探针: "
                f"model={self.emb_model!r}, error={exc}"
            )
            raise EmbeddingProviderUnavailableError(str(exc)) from exc

    def _validate_text_embedding_dimension(
        self, embedding: list[float]
    ) -> list[float]:
        if len(embedding) == self.text_embedding_dimension:
            return embedding
        self._reject_validation(
            "Embedding API 返回维度与当前配置不匹配，拒绝使用向量库: "
            f"当前配置={{'embedding_model': {self.emb_model!r}, "
            f"'embedding_dimension': {self.text_embedding_dimension}}}, "
            f"API返回配置={{'embedding_model': {self.emb_model!r}, "
            f"'embedding_dimension': {len(embedding)}}}；"
            "请修正 embedding_dimension 或更换模型；若已有 collection 按错误维度创建，"
            "请删除并人工重建向量"
        )

    def _embedding_request_kwargs(self, input: list[str]) -> dict[str, Any]:
        """构造 embedding 请求参数；配置了 embedding_dimension 才携带。

        未配置时请求不带 dimensions，使用模型默认输出维度（兼容不支持
        dimensions 参数的 provider，如硅基流动的 BAAI/bge-m3）。
        """
        kwargs: dict[str, Any] = {"input": input, "model": self.emb_model}
        if self.configured_embedding_dimension is not None:
            kwargs["dimensions"] = self.configured_embedding_dimension
        return kwargs

    async def _get_text_embedding(self, text: str) -> list[float] | None:
        """调用 API 获取配置的文本 Dense 向量。"""
        self._raise_if_validation_rejected()
        try:
            resp = await self.emb_client.embeddings.create(
                **self._embedding_request_kwargs([text]),
            )
            return self._validate_text_embedding_dimension(resp.data[0].embedding)
        except CollectionEmbeddingConfigMismatchError:
            raise
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


    async def _rerank(
        self,
        query: str,
        docs: list[str],
        *,
        limit: int = CHAT_SEARCH_RESULT_LIMIT,
    ) -> list[str]:
        """可选地调用 Rerank API；未配置或失败时保留本地融合排序。"""
        if not docs:
            return []

        docs = docs[:CHAT_RERANK_POOL_SIZE]
        if len(docs) == 1 or not str(getattr(self, "rerank_url", "")).strip():
            return docs[:limit]

        try:
            headers = {
                "Content-Type": "application/json"
            }
            rerank_key = str(getattr(self, "rerank_key", "")).strip()
            if rerank_key:
                headers["Authorization"] = f"Bearer {rerank_key}"
            payload = {
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": docs,
                "top_n": min(limit, len(docs)),
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.rerank_url, json=payload, headers=headers)
                resp.raise_for_status()
                results = resp.json().get("results", [])

                ranked: list[str] = []
                seen_indexes: set[int] = set()
                for item in sorted(
                    results,
                    key=lambda value: float(value.get("relevance_score", 0.0)),
                    reverse=True,
                ):
                    index = item.get("index")
                    if (
                        isinstance(index, int)
                        and 0 <= index < len(docs)
                        and index not in seen_indexes
                    ):
                        ranked.append(docs[index])
                        seen_indexes.add(index)
                for index, doc in enumerate(docs):
                    if len(ranked) >= limit:
                        break
                    if index not in seen_indexes:
                        ranked.append(doc)
                return ranked[:limit]
        except Exception as e:
            logger.warning(f"Rerank API 不可用，使用本地融合排序: {type(e).__name__}")
            return docs[:limit]

    @staticmethod
    def _rank_chat_candidates(query: str, points: Sequence[Any]) -> list[str]:
        """用 dense、关键词和按需时效三条路线做加权 RRF。"""
        records: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        seen_msg_id_sets: list[set[int]] = []
        for dense_rank, point in enumerate(points, start=1):
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                continue
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            normalized_text = _normalize_chat_search_text(text)
            if normalized_text in seen_texts:
                continue
            raw_msg_ids = payload.get("msg_ids", [])
            msg_ids = {
                int(value)
                for value in raw_msg_ids
                if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
            } if isinstance(raw_msg_ids, list) else set()
            if len(msg_ids) >= 2 and any(
                len(existing) >= 2
                and (
                    len(msg_ids & existing)
                    / max(min(len(msg_ids), len(existing)), 1)
                    >= 0.75
                )
                for existing in seen_msg_id_sets
            ):
                continue
            seen_texts.add(normalized_text)
            if msg_ids:
                seen_msg_id_sets.append(msg_ids)
            created_at = payload.get("created_at", 0)
            try:
                timestamp = int(created_at)
            except (TypeError, ValueError):
                timestamp = 0
            records.append({
                "text": text.strip(),
                "normalized_text": normalized_text,
                "dense_rank": dense_rank,
                "created_at": timestamp,
                "terms": _chat_search_terms(text),
            })

        if not records:
            return []

        query_terms = _chat_search_terms(query)
        document_frequency = {
            term: sum(term in record["terms"] for record in records)
            for term in query_terms
        }
        term_weights = {
            term: math.log((len(records) + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        total_query_weight = sum(term_weights.values()) or 1.0
        original_query = query.split("\n检索时间范围：", 1)[0]
        normalized_phrase = _normalize_chat_search_text(original_query)

        lexical_scores: dict[int, float] = {}
        for index, record in enumerate(records):
            overlap = query_terms & record["terms"]
            score = sum(term_weights[term] for term in overlap) / total_query_weight
            if len(normalized_phrase) >= 2 and normalized_phrase in record["normalized_text"]:
                score += 0.5
            lexical_scores[index] = score

        fused_scores = {
            index: 1.0 / (CHAT_RRF_K + int(record["dense_rank"]))
            for index, record in enumerate(records)
        }
        lexical_route = sorted(
            (index for index, score in lexical_scores.items() if score > 0),
            key=lambda index: (
                -lexical_scores[index],
                int(records[index]["dense_rank"]),
            ),
        )
        for lexical_rank, index in enumerate(lexical_route, start=1):
            fused_scores[index] += (
                CHAT_LEXICAL_ROUTE_WEIGHT / (CHAT_RRF_K + lexical_rank)
            )

        if CHAT_RECENCY_QUERY_PATTERN.search(original_query):
            recency_route = sorted(
                range(len(records)),
                key=lambda index: (
                    -int(records[index]["created_at"]),
                    int(records[index]["dense_rank"]),
                ),
            )
            for recency_rank, index in enumerate(recency_route, start=1):
                fused_scores[index] += (
                    CHAT_RECENCY_ROUTE_WEIGHT / (CHAT_RRF_K + recency_rank)
                )

        ranked_indexes = sorted(
            range(len(records)),
            key=lambda index: (
                -fused_scores[index],
                int(records[index]["dense_rank"]),
            ),
        )
        return [str(records[index]["text"]) for index in ranked_indexes]

    @staticmethod
    def _trim_chat_fragment(query: str, text: str) -> str:
        """保留命中行附近的少量上下文，限制 RAG 注入体积。"""
        normalized_text = text.strip()
        if len(normalized_text) <= CHAT_RESULT_FRAGMENT_MAX_CHARS:
            return normalized_text

        lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
        if not lines:
            return normalized_text[:CHAT_RESULT_FRAGMENT_MAX_CHARS].rstrip()
        query_terms = _chat_search_terms(query)
        line_scores = [
            len(query_terms & _chat_search_terms(line))
            for line in lines
        ]
        center = max(range(len(lines)), key=lambda index: line_scores[index])
        selected_indexes = [center]
        radius = 1
        while True:
            candidates = []
            if center - radius >= 0:
                candidates.append(center - radius)
            if center + radius < len(lines):
                candidates.append(center + radius)
            if not candidates:
                break
            proposed = sorted(selected_indexes + candidates)
            proposed_text = "\n".join(lines[index] for index in proposed)
            if len(proposed_text) > CHAT_RESULT_FRAGMENT_MAX_CHARS:
                break
            selected_indexes = proposed
            radius += 1
        fragment = "\n".join(lines[index] for index in sorted(selected_indexes))
        return fragment[:CHAT_RESULT_FRAGMENT_MAX_CHARS].rstrip()

    # ================= 聊天记录功能 (RAG) =================

    @staticmethod
    def _chat_point_id(session_id: str, msg_ids: Sequence[int], text: str) -> str:
        identity = (
            f"{CHAT_INDEX_VERSION}\0{session_id}\0"
            f"{','.join(str(value) for value in msg_ids)}\0{text}"
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    @staticmethod
    def _chat_time_filter(
        session_id: str,
        time_range: tuple[int, int] | None,
    ) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            )
        ]
        if time_range is not None:
            range_start, range_end = time_range
            must.extend([
                models.FieldCondition(
                    key="start_at",
                    range=models.Range(lte=range_end),
                ),
                models.FieldCondition(
                    key="end_at",
                    range=models.Range(gte=range_start),
                ),
            ])
        return models.Filter(must=must)

    async def insert_chat(self, text: str, session_id: str):
        """插入新的聊天记录"""
        await self._ensure_collections({self.chat_col})
        vector = await self._get_text_embedding(text)
        if not vector:
            return

        current_time = int(time.time())
        point_id = self._chat_point_id(session_id, (), text)

        point_vector: Any = vector
        if self.chat_col == CHAT_COLLECTION:
            point_vector = {
                CHAT_DENSE_VECTOR: vector,
                CHAT_SPARSE_VECTOR: _chat_sparse_vector(text),
            }
        await self.client.upsert(
            collection_name=self.chat_col,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=point_vector,
                    payload={
                        "session_id": session_id,
                        "text": text,
                        "created_at": current_time,
                        "start_at": current_time,
                        "end_at": current_time,
                        "msg_ids": [],
                        "index_version": CHAT_INDEX_VERSION,
                    }
                )
            ]
        )

    async def search_chat(self, query: str, session_id: str) -> str:
        """在当前会话中召回并融合排序相关历史片段。"""
        if not self.enabled:
            return "未找到相关历史记录"
        normalized_query = query.strip()
        if not normalized_query:
            return "未找到相关历史记录"
        await self._ensure_collections({self.chat_col})
        expanded_query = expand_chat_search_query(normalized_query)
        time_range = parse_chat_search_time_range(normalized_query)
        vector = await self._get_text_embedding(expanded_query)
        if not vector:
            return "无法连接记忆库"

        query_filter = self._chat_time_filter(session_id, time_range)
        if self.chat_col == CHAT_COLLECTION:
            sparse_query = _chat_sparse_vector(normalized_query)
            prefetch: list[models.Prefetch] = [
                models.Prefetch(
                    query=vector,
                    using=CHAT_DENSE_VECTOR,
                    filter=query_filter,
                    limit=CHAT_SEARCH_CANDIDATE_LIMIT,
                )
            ]
            if sparse_query.indices:
                prefetch.append(models.Prefetch(
                    query=sparse_query,
                    using=CHAT_SPARSE_VECTOR,
                    filter=query_filter,
                    limit=CHAT_SEARCH_CANDIDATE_LIMIT,
                ))
            search_result = await self.client.query_points(
                collection_name=self.chat_col,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=CHAT_SEARCH_CANDIDATE_LIMIT,
                with_payload=True,
            )
        else:
            search_result = await self.client.query_points(
                collection_name=self.chat_col,
                query=vector,
                query_filter=query_filter,
                limit=CHAT_SEARCH_CANDIDATE_LIMIT,
                with_payload=True,
            )

        if not search_result or not search_result.points:
            return "未找到相关历史记录"

        candidates = self._rank_chat_candidates(
            expanded_query,
            search_result.points,
        )
        best_texts = await self._rerank(
            expanded_query,
            candidates,
            limit=CHAT_SEARCH_RESULT_LIMIT,
        )
        if not best_texts:
            return "未找到相关历史记录"
        fragments: list[str] = []
        total_chars = 0
        for text in best_texts:
            fragment = self._trim_chat_fragment(normalized_query, text)
            if not fragment:
                continue
            remaining = CHAT_RESULT_TOTAL_MAX_CHARS - total_chars
            if remaining <= 0:
                break
            fragment = fragment[:remaining].rstrip()
            if fragment:
                fragments.append(fragment)
                total_chars += len(fragment)
        if not fragments:
            return "未找到相关历史记录"
        return "\n\n".join(
            f"【相关历史片段 {index}】\n{text}"
            for index, text in enumerate(fragments, start=1)
        )

    # ================= 表情包功能 (Image Search) =================

    async def insert_media(self, media_id: int, image_url: str, description: str) -> bool:
        """插入新表情包 (新图入库用)"""
        if not self.enabled:
            return False
        if self.text_only:
            await self._ensure_collections({self.media_text_col})
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

        await self._ensure_collections(
            self._primary_media_collection_names(),
            validate_text_embedding=False,
        )
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
        批量调用 API 获取文本向量。

        对大规模回填进行串行限速；触发 429 时只重试当前子批次，
        不会丢弃已经成功的向量并从头重做整个批次。
        """
        if not texts:
            return []
        self._raise_if_validation_rejected()

        batch_lock = getattr(self, "_embedding_batch_lock", None)
        if batch_lock is None:
            batch_lock = asyncio.Lock()
            self._embedding_batch_lock = batch_lock

        all_embeddings: list[list[float]] = []
        async with batch_lock:
            for offset in range(0, len(texts), TEXT_EMBEDDING_BATCH_SIZE):
                chunk = texts[offset: offset + TEXT_EMBEDDING_BATCH_SIZE]

                for attempt in range(TEXT_EMBEDDING_RATE_LIMIT_MAX_RETRIES + 1):
                    last_request_at = getattr(
                        self,
                        "_embedding_last_batch_request_at",
                        0.0,
                    )
                    remaining_interval = (
                        TEXT_EMBEDDING_BATCH_MIN_INTERVAL_SECONDS
                        - (time.monotonic() - last_request_at)
                    )
                    if remaining_interval > 0:
                        await asyncio.sleep(remaining_interval)

                    self._embedding_last_batch_request_at = time.monotonic()
                    try:
                        resp = await self.emb_client.embeddings.create(
                            **self._embedding_request_kwargs(chunk),
                        )
                        break
                    except RateLimitError as exc:
                        if attempt >= TEXT_EMBEDDING_RATE_LIMIT_MAX_RETRIES:
                            message = (
                                "Embedding API 持续限流，已停止当前批次，"
                                "保留消息等待后续重试"
                            )
                            logger.error(f"{message}: {exc}")
                            raise EmbeddingProviderUnavailableError(message) from exc

                        retry_after = self._embedding_rate_limit_retry_delay(
                            exc,
                            attempt,
                        )
                        logger.warning(
                            "Embedding API 触发 TPM 限流，"
                            f"{retry_after:.1f} 秒后仅重试当前子批次 "
                            f"(size={len(chunk)}, "
                            f"attempt={attempt + 1}/"
                            f"{TEXT_EMBEDDING_RATE_LIMIT_MAX_RETRIES})"
                        )
                        await asyncio.sleep(retry_after)
                    except CollectionEmbeddingConfigMismatchError:
                        raise
                    except Exception as exc:
                        logger.error(f"Batch Embedding API Error: {exc}")
                        raise EmbeddingProviderUnavailableError(str(exc)) from exc
                else:  # pragma: no cover - for 循环总会 break 或 raise
                    raise EmbeddingProviderUnavailableError(
                        "Embedding API 子批次重试未完成"
                    )

                if len(resp.data) != len(chunk):
                    message = (
                        "Embedding API 返回向量数量异常: "
                        f"expected={len(chunk)}, actual={len(resp.data)}"
                    )
                    logger.error(message)
                    raise EmbeddingProviderUnavailableError(message)

                for data in resp.data:
                    all_embeddings.append(
                        self._validate_text_embedding_dimension(data.embedding)
                    )

        return all_embeddings

    @staticmethod
    def _embedding_rate_limit_retry_delay(
        error: RateLimitError,
        attempt: int,
    ) -> float:
        """Return provider-directed or exponential 429 backoff delay."""
        fallback = min(
            TEXT_EMBEDDING_RATE_LIMIT_MAX_DELAY_SECONDS,
            TEXT_EMBEDDING_RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** attempt),
        )
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {})
        raw_retry_after = headers.get("retry-after") if headers else None
        if isinstance(raw_retry_after, (str, int, float)):
            try:
                provider_delay = float(raw_retry_after)
            except ValueError:
                provider_delay = 0.0
        else:
            provider_delay = 0.0
        return min(
            TEXT_EMBEDDING_RATE_LIMIT_MAX_DELAY_SECONDS,
            max(fallback, provider_delay),
        )

    async def batch_insert(
        self,
        texts: list[str],
        session_id: str,
        *,
        payloads: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """
        批量插入聊天记录 (用于 utils.py 中的历史数据向量化)
        """
        if not self.enabled:
            return
        await self._ensure_collections({self.chat_col})
        if not texts:
            return

        # 1. 批量获取向量
        try:
            vectors = await self._get_batch_text_embeddings(texts)
        except CollectionEmbeddingConfigMismatchError:
            raise
        except EmbeddingProviderUnavailableError:
            raise
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            raise EmbeddingProviderUnavailableError(str(e)) from e

        if len(vectors) != len(texts):
            message = (
                f"向量数量({len(vectors)})与文本数量({len(texts)})不匹配，"
                "保留消息等待重试"
            )
            logger.error(message)
            raise EmbeddingProviderUnavailableError(message)

        # 2. 构造 Qdrant Points
        points = []
        current_time = int(time.time())

        normalized_payloads = list(payloads or [{} for _ in texts])
        if len(normalized_payloads) != len(texts):
            raise ValueError("聊天向量 payload 数量必须与文本数量一致")

        for text, vector, extra_payload in zip(
            texts,
            vectors,
            normalized_payloads,
        ):
            raw_msg_ids = extra_payload.get("msg_ids", [])
            msg_ids = [
                int(value)
                for value in raw_msg_ids
                if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
            ] if isinstance(raw_msg_ids, Sequence) and not isinstance(raw_msg_ids, str) else []
            start_at = int(extra_payload.get("start_at", current_time))
            end_at = int(extra_payload.get("end_at", start_at))
            point_vector: Any = vector
            if self.chat_col == CHAT_COLLECTION:
                point_vector = {
                    CHAT_DENSE_VECTOR: vector,
                    CHAT_SPARSE_VECTOR: _chat_sparse_vector(text),
                }
            points.append(models.PointStruct(
                id=self._chat_point_id(session_id, msg_ids, text),
                vector=point_vector,
                payload={
                    "session_id": session_id,
                    "text": text,
                    "created_at": end_at,
                    "start_at": start_at,
                    "end_at": end_at,
                    "msg_ids": msg_ids,
                    "participants": list(extra_payload.get("participants", [])),
                    "index_version": CHAT_INDEX_VERSION,
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
        """使用加权 RRF 合并文本与视觉召回路线。"""
        merged: dict[int, float] = {}
        for points, weight in routes:
            for rank, point in enumerate(points, start=1):
                media_id = int(point.id)
                merged[media_id] = merged.get(media_id, 0.0) + (
                    weight / (MEME_RRF_K + rank)
                )
        return sorted(merged.items(), key=lambda item: item[1], reverse=True)

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
        async def query_route(
            using: str,
            *,
            label: str,
        ) -> Sequence[models.ScoredPoint] | None:
            kwargs = {
                "collection_name": self.media_multivector_col,
                "query": vector,
                "limit": limit,
                "with_payload": False,
                "timeout": math.ceil(MEME_QDRANT_ROUTE_TIMEOUT_SECONDS),
                "using": using,
                "search_params": models.SearchParams(
                    hnsw_ef=MEME_QDRANT_HNSW_EF,
                    exact=False,
                    indexed_only=True,
                ),
            }
            try:
                result = await asyncio.wait_for(
                    self.client.query_points(**kwargs),
                    timeout=MEME_QDRANT_ROUTE_TIMEOUT_SECONDS + 1.0,
                )
            except Exception as error:
                error_detail = (
                    f"超过 {MEME_QDRANT_ROUTE_TIMEOUT_SECONDS:.1f}s"
                    if isinstance(error, TimeoutError)
                    else str(error).strip() or repr(error)
                )
                logger.warning(f"{label}检索失败，跳过该路线: {error_detail}")
                return None
            return result.points if result else []

        primary_points = await query_route(vector_name, label="独立媒体向量")
        if primary_points is None:
            # 同一集合的另一条向量路线大概率会遇到相同的存储超时，避免把一次
            # 失败请求串成两次完整超时。
            return []

        routes: list[tuple[Sequence[models.ScoredPoint], float]] = [
            (primary_points, 1.0),
        ]
        if vector_name == MEDIA_TEXT_VECTOR and len(primary_points) < limit:
            # 文本向量通常已能返回完整候选池，此时直接结束，避免每次搜图都被
            # 较大的视觉向量索引拖慢。仅当主路线正常完成但候选不足时才用视觉
            # 向量补齐，保留小集合和稀疏集合下的跨模态召回能力。
            visual_points = await query_route(
                MEDIA_IMAGE_VECTOR,
                label="跨模态视觉向量",
            )
            if visual_points is not None:
                routes.append((visual_points, visual_route_weight))
        return self._merge_weighted_meme_search_routes(routes)

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
        expanded_description = expand_meme_search_terms(description)

        if self.text_only:
            await self._ensure_collections({self.media_text_col})
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

        await self._ensure_collections(
            self._primary_media_collection_names(),
            validate_text_embedding=False,
        )
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
        await self._ensure_collections(
            self._primary_media_collection_names(),
            validate_text_embedding=False,
        )

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
