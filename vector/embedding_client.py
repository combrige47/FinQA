"""千问 text-embedding-v4 API 客户端。

通过 OpenAI 兼容接口调用 DashScope Embedding 服务，替代本地 BGE-M3 模型。

用法:
    from vector.embedding_client import EmbeddingClient
    client = EmbeddingClient()
    output = client.encode(["文本1", "文本2"])
    vectors = output["dense_vecs"]  # np.ndarray, shape (N, 1024)
    token_count = len(client.tokenizer.encode("hello"))
"""

import os
import time
import logging
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Tiktoken 适配器 ──
# 模拟 HuggingFace tokenizer 的 encode/decode/__call__ 接口，
# 确保 MarkdownChunker 和 _safe_truncate_text 无需改动。


class _TiktokenAdapter:
    """tiktoken 适配器，提供与 HuggingFace tokenizer 兼容的接口。"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> list[int]:
        """编码文本为 token id 列表。"""
        return self._enc.encode(text)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """解码 token id 列表为文本。"""
        return self._enc.decode(token_ids)

    def __call__(self, text: str, truncation: bool = False, max_length: int | None = None,
                 add_special_tokens: bool = True, return_attention_mask: bool = False,
                 return_tensors=None, **kwargs) -> dict:
        """模拟 HuggingFace tokenizer 的 __call__ 接口（用于 _safe_truncate_text）。"""
        ids = self._enc.encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        result = {"input_ids": ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(ids)
        return result


# ── Embedding 客户端 ──


class EmbeddingClient:
    """千问 text-embedding-v4 API 客户端。

    通过 OpenAI 兼容接口调用 DashScope，提供与旧 BGEM3FlagModel
    兼容的 encode() 和 tokenizer 接口。

    Parameters
    ----------
    api_key : str | None
        DashScope API Key。默认从环境变量 DASHSCOPE_API_KEY 读取。
    model : str
        模型名称，默认 "text-embedding-v4"。
    dimensions : int
        输出向量维度，默认 1024。
    base_url : str | None
        DashScope 兼容接口地址。默认从环境变量 DASHSCOPE_BASE_URL 读取，
        否则使用默认百炼 endpoint。
    max_retries : int
        API 调用失败最大重试次数。
    batch_size : int
        单次 API 调用的最大文本数（text-embedding-v4 上限为 10）。
    """

    # DashScope 默认 endpoint
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        base_url: str | None = None,
        max_retries: int = 3,
        batch_size: int = 10,
    ):
        self.model_name = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.batch_size = batch_size

        _api_key = api_key or os.environ.get("BAILIAN_API_KEY")
        if not _api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置。请设置环境变量或传入 api_key 参数。"
            )
        _base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL", self.DEFAULT_BASE_URL)

        self._client = OpenAI(api_key=_api_key, base_url=_base_url)
        self._tokenizer: _TiktokenAdapter | None = None

    @property
    def tokenizer(self) -> _TiktokenAdapter:
        """返回 tiktoken 适配器（兼容 HuggingFace tokenizer 接口）。"""
        if self._tokenizer is None:
            self._tokenizer = _TiktokenAdapter("cl100k_base")
        return self._tokenizer

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        return_dense: bool = True,
        return_sparse: bool = False,
    ) -> dict:
        """编码文本列表为向量。

        兼容旧 BGEM3FlagModel.encode() 接口：
        - return_dense / return_sparse 参数保留但 sparse 固定返回空列表
        - batch_size 参数保留但仅作内部批大小参考

        Parameters
        ----------
        texts : list[str]
            待编码文本列表。
        batch_size : int | None
            单批大小，默认使用 self.batch_size (10)。
        return_dense : bool
            是否返回密集向量（固定 True，保留为兼容参数）。
        return_sparse : bool
            是否返回稀疏向量（固定忽略，千问 API 不支持稀疏）。

        Returns
        -------
        dict
            {"dense_vecs": np.ndarray (N, dimensions), "lexical_weights": list[None]}
        """
        if not texts:
            return {"dense_vecs": np.array([]).reshape(0, self.dimensions),
                    "lexical_weights": []}

        bs = batch_size or self.batch_size
        all_vectors = []

        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            vectors = self._call_api(batch)
            all_vectors.extend(vectors)

        result = {
            "dense_vecs": np.array(all_vectors, dtype=np.float32),
            "lexical_weights": [None] * len(texts),  # 千问无稀疏向量
        }
        return result

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """调用 DashScope embedding API，带指数退避重试。

        Parameters
        ----------
        texts : list[str]
            单批文本（最多 batch_size 条）。

        Returns
        -------
        list[list[float]]
            向量列表，每个向量长度为 dimensions。
        """
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                # 按 text_index 排序确保顺序
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                return [d.embedding for d in sorted_data]

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # 可重试的错误类型
                if any(kw in msg for kw in ("rate", "throttl", "limit", "timeout",
                                              "server", "busy", "overload", "503", "502", "429")):
                    if attempt < self.max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            f"Embedding API 调用失败 (attempt {attempt + 1}/{self.max_retries}): "
                            f"{e}，{wait}s 后重试..."
                        )
                        time.sleep(wait)
                        continue
                raise

        raise RuntimeError(
            f"Embedding API 调用失败（已重试 {self.max_retries} 次）: {last_err}"
        )

    def safe_truncate_text(self, text: str, limit: int = 7500) -> str:
        """使用 tiktoken 安全截断文本。

        Parameters
        ----------
        text : str
            原始文本。
        limit : int
            token 上限，默认 7500。

        Returns
        -------
        str
            截断后的文本。
        """
        tokenizer = self.tokenizer
        encoded = tokenizer(text, truncation=True, max_length=limit)
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
