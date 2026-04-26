import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(slots=True)
class PairExample:
    text_a: str
    text_b: str
    label: float


class PairDataset(Dataset):
    def __init__(self, pairs: list[PairExample]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> PairExample:
        return self.pairs[index]


def load_config(project_root: Path, config_path: Path) -> dict[str, Any]:
    return json.loads((project_root / config_path).resolve().read_text(encoding="utf-8"))


def mesh_key(mesh_terms: Any) -> str | None:
    if hasattr(mesh_terms, "tolist"):
        mesh_terms = mesh_terms.tolist()
    if not mesh_terms:
        return None
    first = str(mesh_terms[0]).strip().lower()
    return first or None


def build_positive_pairs(
    docs_df: pd.DataFrame,
    max_title_abstract_pairs: int,
    max_mesh_pairs: int,
    rng: random.Random,
) -> list[PairExample]:
    positives: list[PairExample] = []

    title_abstract_df = docs_df[
        docs_df["title"].fillna("").astype(str).str.strip().ne("")
        & docs_df["abstract"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    title_abstract_rows = title_abstract_df.sample(
        n=min(max_title_abstract_pairs, len(title_abstract_df)),
        random_state=rng.randint(0, 10_000),
    )
    for row in title_abstract_rows.to_dict(orient="records"):
        positives.append(
            PairExample(
                text_a=str(row["title"]),
                text_b=str(row["abstract"]),
                label=1.0,
            )
        )

    docs_df = docs_df.copy()
    docs_df["mesh_key"] = docs_df["mesh_terms"].apply(mesh_key)
    grouped = docs_df.dropna(subset=["mesh_key"]).groupby("mesh_key")
    mesh_pairs_added = 0
    for _, group in grouped:
        rows = group.to_dict(orient="records")
        if len(rows) < 2:
            continue
        rng.shuffle(rows)
        for left, right in zip(rows[::2], rows[1::2], strict=False):
            positives.append(
                PairExample(
                    text_a=str(left["retrieval_text"]),
                    text_b=str(right["retrieval_text"]),
                    label=1.0,
                )
            )
            mesh_pairs_added += 1
            if mesh_pairs_added >= max_mesh_pairs:
                return positives

    return positives


def build_negative_pairs(
    docs_df: pd.DataFrame,
    max_negative_pairs: int,
    rng: random.Random,
) -> list[PairExample]:
    negatives: list[PairExample] = []
    rows = docs_df[["pmid", "retrieval_text"]].to_dict(orient="records")
    if len(rows) < 2:
        return negatives

    for _ in range(max_negative_pairs):
        left = rows[rng.randrange(len(rows))]
        right = rows[rng.randrange(len(rows))]
        if left["pmid"] == right["pmid"]:
            continue
        negatives.append(
            PairExample(
                text_a=str(left["retrieval_text"]),
                text_b=str(right["retrieval_text"]),
                label=-1.0,
            )
        )
    return negatives


def split_pairs(
    pairs: list[PairExample],
    validation_split: float,
    rng: random.Random,
) -> tuple[list[PairExample], list[PairExample]]:
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * validation_split)
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]
    return train_pairs, val_pairs


def collate(batch: list[PairExample]) -> dict[str, list[Any]]:
    return {
        "text_a": [item.text_a for item in batch],
        "text_b": [item.text_b for item in batch],
        "labels": [item.label for item in batch],
    }


def move_features_to_device(features: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in features.items()
    }


def evaluate_epoch(model, loader, device) -> float:
    loss_fn = nn.CosineEmbeddingLoss()
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            features_a = model.tokenize(batch["text_a"])
            features_b = model.tokenize(batch["text_b"])
            features_a = move_features_to_device(features_a, device)
            features_b = move_features_to_device(features_b, device)
            labels = torch.tensor(batch["labels"], dtype=torch.float32, device=device)

            emb_a = model(features_a)["sentence_embedding"]
            emb_b = model(features_b)["sentence_embedding"]
            loss = loss_fn(emb_a, emb_b, labels)
            losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root, Path("eval/configs/dense_adaptation_config.json"))

    rng = random.Random(int(config["random_seed"]))
    torch.manual_seed(int(config["random_seed"]))

    docs_df = pd.read_parquet((project_root / config["docs_path"]).resolve())

    positive_pairs = build_positive_pairs(
        docs_df=docs_df,
        max_title_abstract_pairs=int(config["max_title_abstract_pairs"]),
        max_mesh_pairs=int(config["max_mesh_pairs"]),
        rng=rng,
    )
    negative_pairs = build_negative_pairs(
        docs_df=docs_df,
        max_negative_pairs=int(config["max_negative_pairs"]),
        rng=rng,
    )
    all_pairs = positive_pairs + negative_pairs
    train_pairs, val_pairs = split_pairs(
        pairs=all_pairs,
        validation_split=float(config["validation_split"]),
        rng=rng,
    )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config["base_model_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    loss_fn = nn.CosineEmbeddingLoss()

    checkpoint_dir = (project_root / config["checkpoint_dir"]).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir = (project_root / config["output_model_dir"]).resolve()
    output_model_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {
        "config": config,
        "num_positive_pairs": len(positive_pairs),
        "num_negative_pairs": len(negative_pairs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "epochs": [],
    }

    print("Stage 7 training setup")
    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Negative pairs: {len(negative_pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Device: {device}")

    for epoch in range(int(config["epochs"])):
        model.train()
        train_losses: list[float] = []
        total_steps = len(train_loader)
        log_every_steps = int(config.get("log_every_steps", 50))
        for step_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            features_a = model.tokenize(batch["text_a"])
            features_b = model.tokenize(batch["text_b"])
            features_a = move_features_to_device(features_a, device)
            features_b = move_features_to_device(features_b, device)
            labels = torch.tensor(batch["labels"], dtype=torch.float32, device=device)

            emb_a = model(features_a)["sentence_embedding"]
            emb_b = model(features_b)["sentence_embedding"]
            loss = loss_fn(emb_a, emb_b, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

            if log_every_steps > 0 and (
                step_index % log_every_steps == 0 or step_index == total_steps
            ):
                running_loss = sum(train_losses) / max(len(train_losses), 1)
                print(
                    f"Epoch {epoch + 1} step {step_index}/{total_steps}"
                    f" train_loss={running_loss:.4f}",
                    flush=True,
                )

        val_loss = evaluate_epoch(model, val_loader, device)
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "val_loss": val_loss,
        }
        history["epochs"].append(epoch_summary)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}"
        model.save(str(checkpoint_path))
        print(f"Epoch {epoch + 1}: train_loss={epoch_summary['train_loss']:.4f} val_loss={val_loss:.4f}")

    model.save(str(output_model_dir))
    history_path = (project_root / config["history_path"]).resolve()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Stage 7 dense adaptation training complete")
    print(f"Final model: {output_model_dir}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
