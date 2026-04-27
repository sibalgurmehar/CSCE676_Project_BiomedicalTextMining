from __future__ import annotations

import logging
import os
from importlib import metadata

import numpy as np

from app.retrieval.base import BaseRetriever
from app.retrieval.storage import get_method_index_dir, load_dataset_docs, read_json, write_json
from app.retrieval.utils import build_results

LOGGER = logging.getLogger("bioseek.dense_retriever")


class DenseRetriever(BaseRetriever):
    method_name = "dense"

    def __init__(
        self,
        dataset_name: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_name: str | None = None,
    ) -> None:
        super().__init__(dataset_name)
        self.model_name = model_name
        self.index_name = index_name or self.method_name
        self.docs_df = None
        self.model = None
        self.index = None
        self._loaded = False

    @staticmethod
    def _is_offline_mode() -> bool:
        return os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

    @classmethod
    def _sentence_transformer_kwargs(cls) -> list[dict[str, object]]:
        # Prefer the simplest CPU load path first. Older sentence-transformers
        # releases combined with newer transformers versions can trip a meta
        # tensor path when extra model kwargs are forwarded.
        base_kwargs: dict[str, object] = {
            "local_files_only": cls._is_offline_mode(),
            "device": "cpu",
            "backend": "torch",
        }
        eager_kwargs = {
            **base_kwargs,
            "model_kwargs": {
                "low_cpu_mem_usage": False,
                "device_map": None,
            },
        }
        return [base_kwargs, eager_kwargs]

    def _import_dense_dependencies(self):
        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            import faiss
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires 'sentence-transformers' and 'faiss-cpu' to be installed."
            ) from exc
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                LOGGER.warning("Could not set torch interop threads; continuing with existing runtime setting.")
        return faiss, SentenceTransformer

    def _load_sentence_transformer(self, SentenceTransformer, model_name: str):
        last_error: Exception | None = None
        for kwargs in self._sentence_transformer_kwargs():
            try:
                LOGGER.info(
                    "Loading sentence-transformer model=%s offline=%s kwargs=%s",
                    model_name,
                    kwargs["local_files_only"],
                    sorted(key for key in kwargs if key != "model_kwargs"),
                )
                return SentenceTransformer(model_name, **kwargs)
            except Exception as exc:
                last_error = exc
                if "meta tensor" not in str(exc).lower():
                    continue
                LOGGER.warning(
                    "Dense model load hit a meta-tensor path for model=%s with kwargs=%s",
                    model_name,
                    kwargs,
                )

        sentence_transformers_version = metadata.version("sentence-transformers")
        transformers_version = metadata.version("transformers")
        message = (
            "Dense retrieval model loading failed. "
            f"Installed versions: sentence-transformers=={sentence_transformers_version}, "
            f"transformers=={transformers_version}. "
        )
        if last_error and "meta tensor" in str(last_error).lower():
            message += (
                "This usually means the sentence-transformers and transformers "
                "packages are on an incompatible combination that leaves the model "
                "on the meta device during CPU transfer. Reinstall the backend "
                "dependencies from backend/requirements.txt and restart the API."
            )
        elif last_error and "local_files_only" in str(last_error).lower():
            message += (
                "The environment is in offline mode and the embedding model is not "
                "available in the local Hugging Face cache."
            )
        elif last_error:
            message += str(last_error)
        raise RuntimeError(message) from last_error

    def build(self) -> None:
        faiss, SentenceTransformer = self._import_dense_dependencies()
        docs_df = load_dataset_docs(self.dataset_name)
        texts = docs_df["retrieval_text"].fillna("").astype(str).tolist()

        LOGGER.info("Building dense index for dataset=%s model=%s", self.dataset_name, self.model_name)
        model = self._load_sentence_transformer(SentenceTransformer, self.model_name)
        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        index_dir = get_method_index_dir(self.dataset_name, self.index_name)
        docs_df.to_parquet(index_dir / "docs.parquet", index=False)
        np.save(index_dir / "embeddings.npy", embeddings)
        faiss.write_index(index, str(index_dir / "faiss.index"))
        write_json(
            index_dir / "meta.json",
            {
                "dataset_name": self.dataset_name,
                "method": self.index_name,
                "model_name": self.model_name,
                "num_docs": len(docs_df),
                "embedding_dim": embeddings.shape[1],
            },
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        faiss, SentenceTransformer = self._import_dense_dependencies()
        index_dir = get_method_index_dir(self.dataset_name, self.index_name)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            self.build()

        meta = read_json(meta_path)
        self.docs_df = load_dataset_docs(self.dataset_name)
        LOGGER.info(
            "Loading dense retriever model for dataset=%s index=%s from %s",
            self.dataset_name,
            self.index_name,
            meta["model_name"],
        )
        self.model = self._load_sentence_transformer(SentenceTransformer, meta["model_name"])
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self._loaded = True

    def search(self, query: str, top_k: int) -> list:
        self._ensure_loaded()
        LOGGER.info(
            "Running dense search dataset=%s index=%s top_k=%s",
            self.dataset_name,
            self.index_name,
            top_k,
        )
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        filtered = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0], strict=False)
            if idx >= 0
        ]
        return build_results(
            docs_df=self.docs_df,
            indices=[idx for idx, _ in filtered],
            scores=[score for _, score in filtered],
            method=self.index_name,
            dataset_name=self.dataset_name,
        )
