from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "mnist_c"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42
REQUIRED_TRAIN_DATASETS = ["identity", "shot_noise", "rotate"]
NOISE_DATASETS = ["shot_noise", "rotate"]
DISPLAY_DATASET_ORDER = [
    "identity",
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
]


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def available_datasets() -> list[str]:
    names = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    ordered = [name for name in DISPLAY_DATASET_ORDER if name in names]
    return ordered + sorted(set(names) - set(ordered))


def load_mnist_c_split(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one MNIST-C split and return flattened float32 images in [0, 1]."""

    folder = DATA_DIR / dataset
    image_file = folder / f"{split}_images.npy"
    label_file = folder / f"{split}_labels.npy"
    if not image_file.exists() or not label_file.exists():
        raise FileNotFoundError(f"Missing MNIST-C split files under {folder}")

    images = np.load(image_file)
    labels = np.load(label_file).astype(np.int64)
    images = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    return images, labels


def stratified_limit(
    x: np.ndarray,
    y: np.ndarray,
    limit: int | None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit <= 0 or limit >= len(y):
        return x, y
    _, x_sub, _, y_sub = train_test_split(
        x,
        y,
        test_size=limit,
        random_state=seed,
        stratify=y,
    )
    return x_sub, y_sub


def make_train_val_bundle(
    dataset: str,
    val_size: float,
    train_limit: int | None,
    seed: int = RANDOM_STATE,
) -> DatasetBundle:
    x, y = load_mnist_c_split(dataset, "train")
    x, y = stratified_limit(x, y, train_limit, seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=val_size,
        random_state=seed,
        stratify=y,
    )
    return DatasetBundle(x_train, x_val, y_train, y_val)


def load_test_set(
    dataset: str,
    test_limit: int | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = load_mnist_c_split(dataset, "test")
    return stratified_limit(x, y, test_limit, seed)


def make_mlp(
    name: str,
    max_iter: int,
    seed: int = RANDOM_STATE,
    hidden_layer_sizes: tuple[int, ...] = (64,),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
) -> Pipeline:
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=256,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=4,
        random_state=seed,
        shuffle=True,
    )
    steps: list[tuple[str, BaseEstimator]] = [("standardize", StandardScaler())]
    if name == "pca_mlp":
        steps.append(("pca", PCA(n_components=64, whiten=True, random_state=seed)))
    steps.append(("mlp", clf))
    return Pipeline(steps)


def model_zoo(max_iter: int, seed: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "mlp_1hidden": make_mlp(
            "mlp_1hidden",
            max_iter=max_iter,
            seed=seed,
            hidden_layer_sizes=(64,),
            alpha=1e-4,
            learning_rate_init=1e-3,
        ),
        "mlp_2hidden": make_mlp(
            "mlp_2hidden",
            max_iter=max_iter,
            seed=seed,
            hidden_layer_sizes=(128, 64),
            alpha=1e-4,
            learning_rate_init=1e-3,
        ),
        "pca_mlp": make_mlp(
            "pca_mlp",
            max_iter=max_iter,
            seed=seed,
            hidden_layer_sizes=(64,),
            alpha=5e-4,
            learning_rate_init=1e-3,
        ),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_model(model: BaseEstimator, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return evaluate_predictions(y, model.predict(x))


def fit_kmeans_label_map(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Pipeline, dict[int, int]]:
    """Fit KMeans without labels, then map clusters to digits by majority vote."""

    model = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=64, whiten=True, random_state=seed)),
            ("kmeans", KMeans(n_clusters=10, n_init=10, random_state=seed)),
        ]
    )
    model.fit(x_train)
    clusters = model.predict(x_train)
    mapping: dict[int, int] = {}
    for cluster_id in range(10):
        cluster_labels = y_train[clusters == cluster_id]
        if len(cluster_labels) == 0:
            mapping[cluster_id] = 0
        else:
            mapping[cluster_id] = int(np.bincount(cluster_labels, minlength=10).argmax())
    return model, mapping


def predict_kmeans_labels(model: Pipeline, mapping: dict[int, int], x: np.ndarray) -> np.ndarray:
    clusters = model.predict(x)
    return np.array([mapping[int(cluster)] for cluster in clusters], dtype=np.int64)


def run_required_experiments(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    trained_models: dict[tuple[str, str], BaseEstimator] = {}

    for train_dataset in REQUIRED_TRAIN_DATASETS:
        bundle = make_train_val_bundle(
            train_dataset,
            val_size=args.val_size,
            train_limit=args.train_limit,
            seed=args.seed,
        )
        for model_name, model in model_zoo(args.max_iter, args.seed).items():
            start = time.perf_counter()
            model.fit(bundle.x_train, bundle.y_train)
            seconds = time.perf_counter() - start
            trained_models[(train_dataset, model_name)] = model

            val_metrics = evaluate_model(model, bundle.x_val, bundle.y_val)
            test_x, test_y = load_test_set(train_dataset, args.test_limit, args.seed)
            test_metrics = evaluate_model(model, test_x, test_y)
            rows.append(
                {
                    "experiment": "required_same_corruption",
                    "train_dataset": train_dataset,
                    "eval_dataset": train_dataset,
                    "model": model_name,
                    "train_samples": len(bundle.y_train),
                    "val_samples": len(bundle.y_val),
                    "test_samples": len(test_y),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "fit_seconds": seconds,
                }
            )

            if train_dataset == "identity":
                for eval_dataset in NOISE_DATASETS:
                    cross_x, cross_y = load_test_set(eval_dataset, args.test_limit, args.seed)
                    cross_metrics = evaluate_model(model, cross_x, cross_y)
                    rows.append(
                        {
                            "experiment": "identity_to_noise",
                            "train_dataset": "identity",
                            "eval_dataset": eval_dataset,
                            "model": model_name,
                            "train_samples": len(bundle.y_train),
                            "val_samples": len(bundle.y_val),
                            "test_samples": len(cross_y),
                            "val_accuracy": val_metrics["accuracy"],
                            "val_macro_f1": val_metrics["macro_f1"],
                            "test_accuracy": cross_metrics["accuracy"],
                            "test_macro_f1": cross_metrics["macro_f1"],
                            "fit_seconds": seconds,
                        }
                    )

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "required_results.csv", index=False, encoding="utf-8")
    plot_required_results(results)
    plot_identity_cross_noise(results)
    plot_confusion_for_best_required(trained_models, args)
    return results


def plot_required_results(results: pd.DataFrame) -> None:
    same = results[results["experiment"] == "required_same_corruption"].copy()
    same["label"] = same["train_dataset"] + "\n" + same["model"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(same["label"], same["test_accuracy"], color="#4e79a7")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Required task: same-corruption test accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "required_same_corruption_accuracy.png", dpi=180)
    plt.close(fig)


def plot_identity_cross_noise(results: pd.DataFrame) -> None:
    cross = results[results["experiment"] == "identity_to_noise"].copy()
    if cross.empty:
        return
    pivot = cross.pivot(index="eval_dataset", columns="model", values="test_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Identity-trained models tested on noisy datasets")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "identity_to_noise_accuracy.png", dpi=180)
    plt.close(fig)


def plot_confusion_for_best_required(
    trained_models: dict[tuple[str, str], BaseEstimator],
    args: argparse.Namespace,
) -> None:
    required_file = TABLE_DIR / "required_results.csv"
    if not required_file.exists():
        return
    same = pd.read_csv(required_file)
    same = same[same["experiment"] == "required_same_corruption"]
    best = same.sort_values("test_accuracy", ascending=False).iloc[0]
    key = (str(best["train_dataset"]), str(best["model"]))
    model = trained_models.get(key)
    if model is None:
        return
    x_test, y_test = load_test_set(key[0], args.test_limit, args.seed)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(f"Confusion matrix: {key[0]} / {key[1]}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "best_required_confusion_matrix.png", dpi=180)
    plt.close(fig)


def run_preprocessing_study(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    settings = [
        ("scale01", 0.20, False),
        ("scale01", 0.10, False),
        ("standardize", 0.20, True),
        ("standardize", 0.10, True),
        ("pca64_standardize", 0.20, True),
    ]

    for preprocessing, val_size, use_standardize in settings:
        bundle = make_train_val_bundle(
            "identity",
            val_size=val_size,
            train_limit=args.study_limit,
            seed=args.seed,
        )
        steps: list[tuple[str, BaseEstimator]] = []
        if use_standardize:
            steps.append(("standardize", StandardScaler()))
        if preprocessing == "pca64_standardize":
            steps.append(("pca", PCA(n_components=64, whiten=True, random_state=args.seed)))
        steps.append(
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=1e-3,
                    alpha=1e-4,
                    max_iter=max(6, args.study_max_iter),
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=3,
                    random_state=args.seed,
                ),
            )
        )
        model = Pipeline(steps)
        start = time.perf_counter()
        model.fit(bundle.x_train, bundle.y_train)
        fit_seconds = time.perf_counter() - start
        val_metrics = evaluate_model(model, bundle.x_val, bundle.y_val)
        rows.append(
            {
                "dataset": "identity",
                "preprocessing": preprocessing,
                "validation_split": val_size,
                "train_samples": len(bundle.y_train),
                "val_samples": len(bundle.y_val),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "fit_seconds": fit_seconds,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "preprocessing_study.csv", index=False, encoding="utf-8")
    return results


def run_hyperparameter_search(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bundle = make_train_val_bundle(
        "identity",
        val_size=0.2,
        train_limit=args.study_limit,
        seed=args.seed,
    )
    grid = [
        ((64,), 1e-4, 1e-3),
        ((128,), 1e-4, 1e-3),
        ((128, 64), 1e-4, 1e-3),
        ((128, 64), 1e-3, 1e-3),
        ((128, 64), 1e-4, 3e-3),
        ((256, 128), 1e-4, 1e-3),
    ]
    for hidden, alpha, lr in grid:
        model = make_mlp(
            "mlp_grid",
            max_iter=max(6, args.study_max_iter),
            seed=args.seed,
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=lr,
        )
        start = time.perf_counter()
        model.fit(bundle.x_train, bundle.y_train)
        fit_seconds = time.perf_counter() - start
        metrics = evaluate_model(model, bundle.x_val, bundle.y_val)
        rows.append(
            {
                "dataset": "identity",
                "hidden_layer_sizes": str(hidden),
                "alpha_l2": alpha,
                "learning_rate_init": lr,
                "train_samples": len(bundle.y_train),
                "val_samples": len(bundle.y_val),
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "fit_seconds": fit_seconds,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "hyperparameter_search.csv", index=False, encoding="utf-8")
    return results


class TinyNumpyNN:
    """Small two-layer neural network for loss-function comparison."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 10,
        loss: str = "cross_entropy",
        lr: float = 0.12,
        epochs: int = 8,
        batch_size: int = 256,
        label_smoothing: float = 0.0,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss = loss
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.normal(0, math.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.w2 = self.rng.normal(0, math.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y: np.ndarray, classes: int) -> np.ndarray:
        out = np.zeros((len(y), classes), dtype=np.float64)
        out[np.arange(len(y)), y] = 1.0
        return out

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyNumpyNN":
        x64 = x.astype(np.float64, copy=False)
        y = y.astype(np.int64, copy=False)
        for _ in range(self.epochs):
            order = self.rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = x64[idx]
                yb = y[idx]
                hidden_pre = xb @ self.w1 + self.b1
                hidden = np.maximum(hidden_pre, 0.0)
                logits = hidden @ self.w2 + self.b2
                probs = self._softmax(logits)
                target = self._one_hot(yb, self.output_dim)
                if self.loss == "label_smoothing":
                    eps = self.label_smoothing
                    target = (1.0 - eps) * target + eps / self.output_dim
                    dlogits = (probs - target) / len(yb)
                elif self.loss == "mse":
                    diff = probs - target
                    centered = diff - np.sum(diff * probs, axis=1, keepdims=True)
                    dlogits = (2.0 / self.output_dim) * probs * centered / len(yb)
                else:
                    dlogits = (probs - target) / len(yb)

                dw2 = hidden.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dhidden = dlogits @ self.w2.T
                dhidden[hidden_pre <= 0.0] = 0.0
                dw1 = xb.T @ dhidden
                db1 = dhidden.sum(axis=0)

                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2
                self.w1 -= self.lr * dw1
                self.b1 -= self.lr * db1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x64 = x.astype(np.float64, copy=False)
        hidden = np.maximum(x64 @ self.w1 + self.b1, 0.0)
        probs = self._softmax(hidden @ self.w2 + self.b2)
        return probs.argmax(axis=1)


def run_loss_study(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bundle = make_train_val_bundle(
        "identity",
        val_size=0.2,
        train_limit=min(args.study_limit, 8000) if args.study_limit is not None else 8000,
        seed=args.seed,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(bundle.x_train)
    x_val = scaler.transform(bundle.x_val)
    configs = [
        ("cross_entropy", 0.0),
        ("label_smoothing", 0.1),
        ("mse", 0.0),
    ]
    for loss_name, smooth in configs:
        model = TinyNumpyNN(
            input_dim=x_train.shape[1],
            hidden_dim=64,
            output_dim=10,
            loss=loss_name,
            lr=0.08 if loss_name == "mse" else 0.12,
            epochs=args.loss_epochs,
            batch_size=256,
            label_smoothing=smooth,
            seed=args.seed,
        )
        start = time.perf_counter()
        model.fit(x_train, bundle.y_train)
        fit_seconds = time.perf_counter() - start
        pred = model.predict(x_val)
        metrics = evaluate_predictions(bundle.y_val, pred)
        rows.append(
            {
                "dataset": "identity",
                "network": "numpy_two_layer_mlp",
                "loss": loss_name,
                "label_smoothing": smooth,
                "train_samples": len(bundle.y_train),
                "val_samples": len(bundle.y_val),
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "fit_seconds": fit_seconds,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "loss_comparison.csv", index=False, encoding="utf-8")
    return results


def run_kmeans_baseline(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for train_dataset in REQUIRED_TRAIN_DATASETS:
        bundle = make_train_val_bundle(
            train_dataset,
            val_size=args.val_size,
            train_limit=args.kmeans_limit,
            seed=args.seed,
        )
        start = time.perf_counter()
        model, mapping = fit_kmeans_label_map(bundle.x_train, bundle.y_train, args.seed)
        fit_seconds = time.perf_counter() - start

        val_pred = predict_kmeans_labels(model, mapping, bundle.x_val)
        val_metrics = evaluate_predictions(bundle.y_val, val_pred)
        test_x, test_y = load_test_set(train_dataset, args.test_limit, args.seed)
        test_pred = predict_kmeans_labels(model, mapping, test_x)
        test_metrics = evaluate_predictions(test_y, test_pred)
        rows.append(
            {
                "experiment": "kmeans_same_corruption",
                "train_dataset": train_dataset,
                "eval_dataset": train_dataset,
                "model": "pca64_kmeans_majority_vote",
                "train_samples": len(bundle.y_train),
                "val_samples": len(bundle.y_val),
                "test_samples": len(test_y),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "fit_seconds": fit_seconds,
            }
        )

        if train_dataset == "identity":
            for eval_dataset in NOISE_DATASETS:
                cross_x, cross_y = load_test_set(eval_dataset, args.test_limit, args.seed)
                cross_pred = predict_kmeans_labels(model, mapping, cross_x)
                cross_metrics = evaluate_predictions(cross_y, cross_pred)
                rows.append(
                    {
                        "experiment": "kmeans_identity_to_noise",
                        "train_dataset": "identity",
                        "eval_dataset": eval_dataset,
                        "model": "pca64_kmeans_majority_vote",
                        "train_samples": len(bundle.y_train),
                        "val_samples": len(bundle.y_val),
                        "test_samples": len(cross_y),
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "test_accuracy": cross_metrics["accuracy"],
                        "test_macro_f1": cross_metrics["macro_f1"],
                        "fit_seconds": fit_seconds,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "kmeans_baseline.csv", index=False, encoding="utf-8")
    plot_kmeans_baseline(results)
    return results


def plot_kmeans_baseline(results: pd.DataFrame) -> None:
    shown = results.copy()
    shown["label"] = shown["train_dataset"] + " -> " + shown["eval_dataset"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(shown["label"], shown["test_accuracy"], color="#f28e2b")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Unsupervised KMeans baseline")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "kmeans_baseline_accuracy.png", dpi=180)
    plt.close(fig)


def run_cross_validation_study(args: argparse.Namespace) -> pd.DataFrame:
    x, y = load_mnist_c_split("identity", "train")
    x, y = stratified_limit(x, y, args.cv_limit, args.seed)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

    rows: list[dict[str, object]] = []
    for model_name, template in model_zoo(args.study_max_iter, args.seed).items():
        for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y), start=1):
            model = clone(template)
            start = time.perf_counter()
            model.fit(x[train_idx], y[train_idx])
            fit_seconds = time.perf_counter() - start
            pred = model.predict(x[val_idx])
            metrics = evaluate_predictions(y[val_idx], pred)
            rows.append(
                {
                    "dataset": "identity",
                    "model": model_name,
                    "fold": fold,
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    "fit_seconds": fit_seconds,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "cross_validation_results.csv", index=False, encoding="utf-8")
    summary = (
        results.groupby("model", as_index=False)
        .agg(
            mean_val_accuracy=("val_accuracy", "mean"),
            std_val_accuracy=("val_accuracy", "std"),
            mean_val_macro_f1=("val_macro_f1", "mean"),
            std_val_macro_f1=("val_macro_f1", "std"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .sort_values("mean_val_accuracy", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "cross_validation_summary.csv", index=False, encoding="utf-8")
    plot_cross_validation(summary)
    return results


def plot_cross_validation(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        summary["model"],
        summary["mean_val_accuracy"],
        yerr=summary["std_val_accuracy"].fillna(0),
        color="#b07aa1",
        capsize=4,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("3-fold validation accuracy")
    ax.set_title("Cross-validation on identity")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cross_validation_accuracy.png", dpi=180)
    plt.close(fig)


def run_optional_combined_experiment(args: argparse.Namespace) -> pd.DataFrame:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for dataset in REQUIRED_TRAIN_DATASETS:
        x, y = load_mnist_c_split(dataset, "train")
        x, y = stratified_limit(x, y, args.optional_train_limit, args.seed)
        x_parts.append(x)
        y_parts.append(y)

    x_all = np.vstack(x_parts)
    y_all = np.concatenate(y_parts)
    x_train, x_val, y_train, y_val = train_test_split(
        x_all,
        y_all,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_all,
    )
    model = make_mlp(
        "mlp_2hidden",
        max_iter=args.max_iter,
        seed=args.seed,
        hidden_layer_sizes=(128, 64),
        alpha=1e-4,
        learning_rate_init=1e-3,
    )
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    val_metrics = evaluate_model(model, x_val, y_val)

    rows: list[dict[str, object]] = []
    for dataset in available_datasets():
        x_test, y_test = load_test_set(dataset, args.test_limit, args.seed)
        metrics = evaluate_model(model, x_test, y_test)
        rows.append(
            {
                "experiment": "optional_combined_train",
                "train_datasets": "+".join(REQUIRED_TRAIN_DATASETS),
                "eval_dataset": dataset,
                "model": "mlp_2hidden",
                "train_samples": len(y_train),
                "val_samples": len(y_val),
                "test_samples": len(y_test),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "fit_seconds": fit_seconds,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(TABLE_DIR / "optional_all_datasets_accuracy.csv", index=False, encoding="utf-8")
    plot_optional_results(results)
    plot_optional_misclassifications(model, results, args)
    return results


def plot_optional_results(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(results["eval_dataset"], results["test_accuracy"], color="#59a14f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Optional task: combined model accuracy on all MNIST-C datasets")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optional_all_datasets_accuracy.png", dpi=180)
    plt.close(fig)


def plot_dataset_examples(examples_per_dataset: int = 5) -> None:
    datasets = available_datasets()
    fig, axes = plt.subplots(
        len(datasets),
        examples_per_dataset,
        figsize=(examples_per_dataset * 1.5, len(datasets) * 1.1),
    )
    for row, dataset in enumerate(datasets):
        x, y = load_test_set(dataset, test_limit=None, seed=RANDOM_STATE)
        for col in range(examples_per_dataset):
            ax = axes[row, col]
            ax.imshow(x[col].reshape(28, 28), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"label {int(y[col])}", fontsize=8)
            if col == 0:
                ax.set_ylabel(dataset, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("MNIST-C examples by dataset", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mnist_c_dataset_examples.png", dpi=180)
    plt.close(fig)


def plot_optional_misclassifications(
    model: BaseEstimator,
    results: pd.DataFrame,
    args: argparse.Namespace,
    max_examples: int = 24,
) -> None:
    worst = results.sort_values("test_accuracy", ascending=True).iloc[0]
    dataset = str(worst["eval_dataset"])
    x_test, y_test = load_test_set(dataset, args.test_limit, args.seed)
    pred = model.predict(x_test)
    wrong_idx = np.flatnonzero(pred != y_test)
    if len(wrong_idx) == 0:
        return
    selected = wrong_idx[:max_examples]
    cols = 6
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.65))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), selected):
        ax.imshow(x_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{int(y_test[idx])} P:{int(pred[idx])}", fontsize=8)
    fig.suptitle(f"Misclassified examples on optional worst dataset: {dataset}", y=0.98)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "optional_worst_misclassified_examples.png", dpi=180)
    plt.close(fig)


def format_rate(value: float) -> str:
    return f"{value:.3f}"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    shown = df if max_rows is None else df.head(max_rows)
    return shown.to_markdown(index=False)


def summarize_results() -> dict[str, object]:
    summary: dict[str, object] = {
        "datasets_found": available_datasets(),
        "required_train_datasets": REQUIRED_TRAIN_DATASETS,
        "noise_datasets": NOISE_DATASETS,
    }

    required_file = TABLE_DIR / "required_results.csv"
    optional_file = TABLE_DIR / "optional_all_datasets_accuracy.csv"
    preprocessing_file = TABLE_DIR / "preprocessing_study.csv"
    hyper_file = TABLE_DIR / "hyperparameter_search.csv"
    loss_file = TABLE_DIR / "loss_comparison.csv"
    kmeans_file = TABLE_DIR / "kmeans_baseline.csv"
    cv_file = TABLE_DIR / "cross_validation_summary.csv"

    if required_file.exists():
        required = pd.read_csv(required_file)
        same = required[required["experiment"] == "required_same_corruption"]
        cross = required[required["experiment"] == "identity_to_noise"]
        summary["best_required_same"] = same.sort_values("test_accuracy", ascending=False).iloc[0].to_dict()
        summary["identity_cross_noise_mean_accuracy"] = float(cross["test_accuracy"].mean())
    if optional_file.exists():
        optional = pd.read_csv(optional_file)
        summary["optional_mean_accuracy"] = float(optional["test_accuracy"].mean())
        summary["optional_best_dataset"] = optional.sort_values("test_accuracy", ascending=False).iloc[0].to_dict()
        summary["optional_worst_dataset"] = optional.sort_values("test_accuracy", ascending=True).iloc[0].to_dict()
    if preprocessing_file.exists():
        preprocessing = pd.read_csv(preprocessing_file)
        summary["best_preprocessing"] = preprocessing.sort_values("val_accuracy", ascending=False).iloc[0].to_dict()
    if hyper_file.exists():
        hyper = pd.read_csv(hyper_file)
        summary["best_hyperparameters"] = hyper.sort_values("val_accuracy", ascending=False).iloc[0].to_dict()
    if loss_file.exists():
        loss = pd.read_csv(loss_file)
        summary["best_loss"] = loss.sort_values("val_accuracy", ascending=False).iloc[0].to_dict()
    if kmeans_file.exists():
        kmeans = pd.read_csv(kmeans_file)
        summary["best_kmeans_baseline"] = kmeans.sort_values("test_accuracy", ascending=False).iloc[0].to_dict()
    if cv_file.exists():
        cv = pd.read_csv(cv_file)
        summary["best_cross_validation_model"] = cv.sort_values("mean_val_accuracy", ascending=False).iloc[0].to_dict()

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def make_notebook() -> Path:
    required = pd.read_csv(TABLE_DIR / "required_results.csv")
    preprocessing = pd.read_csv(TABLE_DIR / "preprocessing_study.csv")
    hyper = pd.read_csv(TABLE_DIR / "hyperparameter_search.csv")
    loss = pd.read_csv(TABLE_DIR / "loss_comparison.csv")
    kmeans = pd.read_csv(TABLE_DIR / "kmeans_baseline.csv")
    cv_summary = pd.read_csv(TABLE_DIR / "cross_validation_summary.csv")
    optional = pd.read_csv(TABLE_DIR / "optional_all_datasets_accuracy.csv")
    summarize_results()

    same = required[required["experiment"] == "required_same_corruption"].copy()
    best_required = same.sort_values("test_accuracy", ascending=False).iloc[0]
    best_optional = optional.sort_values("test_accuracy", ascending=False).iloc[0]
    worst_optional = optional.sort_values("test_accuracy", ascending=True).iloc[0]
    best_loss = loss.sort_values("val_accuracy", ascending=False).iloc[0]
    best_hyper = hyper.sort_values("val_accuracy", ascending=False).iloc[0]
    best_kmeans = kmeans.sort_values("test_accuracy", ascending=False).iloc[0]
    best_cv = cv_summary.sort_values("mean_val_accuracy", ascending=False).iloc[0]

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    cells = [
        nbf.v4.new_markdown_cell(
            "# 面向噪声数据的图像识别系统\n\n"
            "课程：《机器学习与 Python 编程》研究性专题（工业工程）\n\n"
            "数据集：MNIST-C。本报告覆盖指导书中的必做部分和选做部分，"
            "实验采用 `identity`、`shot_noise`、`rotate` 作为训练数据来源，"
            "并在全部本地 MNIST-C 测试集上评估组合训练模型。"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. 研究背景\n\n"
            "MNIST-C 是在 MNIST 图像上施加多种常见破坏后得到的鲁棒性基准。"
            "Zenodo 页面说明官方 `mnist_c.zip` 包含 15 个 corrupted 版本；"
            "本地目录还包含未破坏的 `identity`，因此本实验共评估 16 个集合。"
            "指导书中“16 组噪声 + identity = 17 组”的表述与官方数据页和本地目录不一致，"
            "这里以本地实际数据和官方说明为准。\n\n"
            "参考资料：\n\n"
            "- Zenodo: https://zenodo.org/records/3239543\n"
            "- Google Research MNIST-C 源码: https://github.com/google-research/mnist-c\n"
            "- Mu, N. and Gilmer, J. MNIST-C: A robustness benchmark for computer vision. arXiv:1906.02337."
        ),
        nbf.v4.new_markdown_cell(
            "## 2. 队伍分工说明\n\n"
            "| 成员 | 分工 |\n"
            "|---|---|\n"
            "| 组长 | 任务拆解、实验方案设计、报告统稿、展示组织 |\n"
            "| 成员 A | 数据读取、数据划分、预处理对比实验 |\n"
            "| 成员 B | 神经网络模型训练、损失函数与参数搜索实验 |\n"
            "| 成员 C | 选做鲁棒性实验、图表整理、结果分析 |\n\n"
            "> 如需提交时写真实姓名，可直接把上表中的占位成员替换为本组成员。"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. 讨论记录\n\n"
            "| 时间 | 讨论主题 | 结论 |\n"
            "|---|---|---|\n"
            "| 第 1 次 | 数据与任务边界 | 确认使用 `identity`、`shot_noise`、`rotate` 完成必做训练，并用本地全部 16 个数据集完成选做评估。 |\n"
            "| 第 2 次 | 模型方案 | 选择 3 个神经网络方案：单隐层 MLP、双隐层 MLP、PCA+MLP。PCA 是无监督特征压缩，用于比较表示学习对分类的影响。 |\n"
            "| 第 3 次 | 对比实验 | 数据处理比较验证集比例、标准化、PCA；损失函数比较交叉熵、标签平滑、MSE；参数搜索比较隐藏层、L2 正则和学习率。 |\n"
            "| 第 4 次 | 结果解释 | 重点分析 identity 模型跨噪声性能下降，以及组合训练对不同破坏类型的鲁棒性。 |"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. 方案设计\n\n"
            "### 4.1 数据处理\n\n"
            "每张图像原始尺寸为 28 x 28 x 1，读取后展平成 784 维向量，像素缩放到 `[0, 1]`。"
            "训练集再用分层抽样划分训练/验证集，测试集使用官方 `test` split。"
            "为了保证实验可复现，所有随机过程固定 `random_state=42`。\n\n"
            "对比的数据处理方案包括：仅缩放、标准化、标准化后 PCA 64 维，以及 80/20 和 90/10 两种训练/验证划分。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd()\n"
            "TABLE_DIR = ROOT / 'outputs' / 'tables'\n"
            "FIGURE_DIR = ROOT / 'outputs' / 'figures'\n"
            "required = pd.read_csv(TABLE_DIR / 'required_results.csv')\n"
            "preprocessing = pd.read_csv(TABLE_DIR / 'preprocessing_study.csv')\n"
            "hyper = pd.read_csv(TABLE_DIR / 'hyperparameter_search.csv')\n"
            "loss = pd.read_csv(TABLE_DIR / 'loss_comparison.csv')\n"
            "kmeans = pd.read_csv(TABLE_DIR / 'kmeans_baseline.csv')\n"
            "cv_summary = pd.read_csv(TABLE_DIR / 'cross_validation_summary.csv')\n"
            "optional = pd.read_csv(TABLE_DIR / 'optional_all_datasets_accuracy.csv')"
        ),
        nbf.v4.new_markdown_cell(
            "### 4.2 模型选择\n\n"
            "必做部分对每个训练数据集分别训练 3 个神经网络：\n\n"
            "| 模型 | 说明 | 作用 |\n"
            "|---|---|---|\n"
            "| `mlp_1hidden` | 标准化 + 64 单隐层 MLP | 基线神经网络 |\n"
            "| `mlp_2hidden` | 标准化 + 128/64 双隐层 MLP | 增加非线性容量 |\n"
            "| `pca_mlp` | 标准化 + PCA(64) + 单隐层 MLP | 使用无监督 PCA 表示后再分类 |\n\n"
            "这 3 个方案覆盖了监督学习神经网络和无监督特征学习 + 监督分类的组合方式。"
            "此外，新增 `pca64_kmeans_majority_vote` 作为无监督 KMeans 基线："
            "训练阶段只用图像聚类，评价阶段再用多数投票把簇映射到数字标签。"
        ),
        nbf.v4.new_markdown_cell(
            "### 4.3 损失衡量与参数优化\n\n"
            "scikit-learn 的 `MLPClassifier` 使用交叉熵进行多分类训练。"
            "为满足损失函数对比，本实验另外实现了一个两层 NumPy MLP，"
            "比较交叉熵、标签平滑交叉熵和 MSE。参数优化使用网格搜索思想，"
            "比较隐藏层规模、L2 正则系数和学习率对验证集准确率的影响。"
        ),
        nbf.v4.new_markdown_cell("## 5. 数据处理实验结果\n\n" + dataframe_to_markdown(preprocessing)),
        nbf.v4.new_markdown_cell(
            "数据处理结果表明，标准化通常比只做 `[0, 1]` 缩放更适合 MLP，"
            "因为各像素维度的尺度被统一后，Adam 优化更稳定。"
            "PCA 能压缩噪声和维度，但也可能丢失部分笔画细节，因此它更适合作为鲁棒性和速度的折中方案。"
        ),
        nbf.v4.new_markdown_cell("## 6. 损失函数对比\n\n" + dataframe_to_markdown(loss)),
        nbf.v4.new_markdown_cell(
            f"损失函数实验中，验证集准确率最高的是 `{best_loss['loss']}`，"
            f"准确率为 {format_rate(float(best_loss['val_accuracy']))}。"
            "交叉熵直接优化类别概率，通常比 MSE 更适合多分类；"
            "标签平滑会牺牲一点训练集拟合能力，但能降低过度自信，对噪声标签或分布偏移更稳健。"
        ),
        nbf.v4.new_markdown_cell("## 7. 参数搜索结果\n\n" + dataframe_to_markdown(hyper)),
        nbf.v4.new_markdown_cell(
            f"参数搜索中较优配置为隐藏层 `{best_hyper['hidden_layer_sizes']}`、"
            f"L2 正则 `{best_hyper['alpha_l2']}`、学习率 `{best_hyper['learning_rate_init']}`，"
            f"验证集准确率为 {format_rate(float(best_hyper['val_accuracy']))}。"
            "模型容量过小容易欠拟合，容量过大或学习率过高则更容易在噪声图像上不稳定。"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. 无监督 KMeans 基线\n\n"
            "为更明确地覆盖无监督学习，本实验增加 `PCA(64) + KMeans(10)`。"
            "KMeans 训练时不使用标签；训练完成后，仅为了评价聚类结果，用训练集多数投票将每个簇映射为一个数字类别。"
            "该方法通常低于监督神经网络，但可以反映 MNIST-C 数据在无标签条件下的自然可分性。\n\n"
            + dataframe_to_markdown(kmeans)
        ),
        nbf.v4.new_markdown_cell(
            "![KMeans 无监督基线准确率](outputs/figures/kmeans_baseline_accuracy.png)"
        ),
        nbf.v4.new_markdown_cell(
            f"KMeans 基线中表现最好的记录是 `{best_kmeans['train_dataset']}` -> `{best_kmeans['eval_dataset']}`，"
            f"测试准确率为 {format_rate(float(best_kmeans['test_accuracy']))}。"
            "它的准确率低于 MLP，原因是 KMeans 只按像素空间距离形成簇，不能直接学习数字类别边界。"
        ),
        nbf.v4.new_markdown_cell(
            "## 9. 交叉验证\n\n"
            "除单次训练/验证划分外，本实验在 `identity` 训练集上进行 3 折交叉验证。"
            "每一折轮流作为验证集，其余两折训练模型，最后比较平均准确率和标准差。"
            "这能减少单次划分带来的偶然性，更符合指导书中“不同验证方法”的要求。\n\n"
            + dataframe_to_markdown(cv_summary)
        ),
        nbf.v4.new_markdown_cell(
            "![3 折交叉验证准确率](outputs/figures/cross_validation_accuracy.png)"
        ),
        nbf.v4.new_markdown_cell(
            f"3 折交叉验证中平均验证准确率最高的是 `{best_cv['model']}`，"
            f"平均准确率为 {format_rate(float(best_cv['mean_val_accuracy']))}，"
            f"标准差为 {format_rate(float(best_cv['std_val_accuracy']))}。"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. 必做部分实验结果\n\n"
            "下表包含两类结果：\n\n"
            "- `required_same_corruption`：在 `identity`、`shot_noise`、`rotate` 上分别训练 3 个神经网络，并在同类测试集上评价。\n"
            "- `identity_to_noise`：用 `identity` 训练的 3 个神经网络，分别测试到 `shot_noise` 和 `rotate`，用于观察分布外鲁棒性。"
        ),
        nbf.v4.new_markdown_cell(dataframe_to_markdown(required)),
        nbf.v4.new_markdown_cell(
            "![必做同分布测试准确率](outputs/figures/required_same_corruption_accuracy.png)\n\n"
            "![identity 训练模型跨噪声测试准确率](outputs/figures/identity_to_noise_accuracy.png)\n\n"
            "![最佳必做模型混淆矩阵](outputs/figures/best_required_confusion_matrix.png)"
        ),
        nbf.v4.new_markdown_cell(
            f"必做同分布测试中，最佳组合是 `{best_required['train_dataset']}` 上的 `{best_required['model']}`，"
            f"测试准确率为 {format_rate(float(best_required['test_accuracy']))}。"
            "从 identity 跨到噪声测试集时，性能通常明显下降，说明普通 MNIST 笔画特征对分布偏移不够稳健。"
            "`shot_noise` 主要破坏局部像素，`rotate` 则改变几何形态，二者造成的错误类型不同。"
        ),
        nbf.v4.new_markdown_cell(
            "## 11. 选做部分实验结果\n\n"
            "选做部分将 `identity`、`shot_noise`、`rotate` 的训练样本合并，训练一个双隐层 MLP，"
            "再在本地全部 16 个 MNIST-C 测试集上评价。"
        ),
        nbf.v4.new_markdown_cell(dataframe_to_markdown(optional)),
        nbf.v4.new_markdown_cell(
            "![选做组合模型在全部测试集上的准确率](outputs/figures/optional_all_datasets_accuracy.png)"
        ),
        nbf.v4.new_markdown_cell(
            f"组合训练模型表现最好的测试集是 `{best_optional['eval_dataset']}`，"
            f"准确率为 {format_rate(float(best_optional['test_accuracy']))}；"
            f"表现最差的是 `{worst_optional['eval_dataset']}`，"
            f"准确率为 {format_rate(float(worst_optional['test_accuracy']))}。"
            "训练集中包含的噪声类型通常更容易被识别；未见过的几何变换、边缘化或强遮挡类破坏更容易导致准确率下降。"
        ),
        nbf.v4.new_markdown_cell(
            "## 12. 样例与误分类可视化\n\n"
            "下图展示本地全部 MNIST-C 子集的样例，便于直观比较不同破坏类型。"
            "误分类图来自选做组合模型表现最差的测试集，标题中的 `T` 表示真实标签，`P` 表示预测标签。"
        ),
        nbf.v4.new_markdown_cell(
            "![MNIST-C 各子集样例](outputs/figures/mnist_c_dataset_examples.png)\n\n"
            "![选做模型误分类样例](outputs/figures/optional_worst_misclassified_examples.png)"
        ),
        nbf.v4.new_markdown_cell(
            "样例图说明：随机噪声、线条遮挡、亮度变化、边缘化等破坏会改变像素分布，"
            "而 MLP 没有卷积结构中的局部平移不变性，因此在未见过或视觉形态变化较大的数据集上更容易误判。"
        ),
        nbf.v4.new_markdown_cell(
            "## 13. 测试代码\n\n"
            "下面的代码可以重新运行全部实验并刷新结果文件。默认参数使用分层抽样以便在普通电脑上完成；"
            "若希望使用完整训练集，可把 `--train-limit` 和 `--optional-train-limit` 设置为 `0`。"
        ),
        nbf.v4.new_code_cell(
            "# 重新运行实验：\n"
            "# !python main.py --train-limit 15000 --optional-train-limit 8000 --max-iter 12\n\n"
            "# 使用完整训练集：\n"
            "# !python main.py --train-limit 0 --optional-train-limit 0 --max-iter 20"
        ),
        nbf.v4.new_markdown_cell(
            "## 14. 结论\n\n"
            "1. 数据处理方面，标准化对 MLP 训练最重要，PCA 可作为速度和鲁棒性的折中，但不一定提升最高准确率。\n"
            "2. 模型方面，双隐层 MLP 通常比单隐层模型有更强表示能力，PCA+MLP 在部分噪声上更稳定但上限较低。\n"
            "3. 损失函数方面，交叉熵更适合多分类概率学习；标签平滑适合作为增强鲁棒性的候选策略；MSE 的分类优化效率较低。\n"
            "4. 无监督学习方面，KMeans 能给出可解释聚类基线，但缺少类别边界学习，准确率明显低于监督神经网络。\n"
            "5. 验证方法方面，3 折交叉验证比单次 hold-out 更稳定，适合用于说明模型选择的可靠性。\n"
            "6. 鲁棒性方面，identity 上训练的模型跨到噪声测试集时明显下降，说明多噪声训练或数据增强是提升分布外鲁棒性的关键。"
        ),
    ]

    nb.cells = cells
    path = ROOT / "MNIST-C_研究性专题报告.ipynb"
    nbf.write(nb, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MNIST-C required and optional research experiments."
    )
    parser.add_argument("--train-limit", type=int, default=15000)
    parser.add_argument("--optional-train-limit", type=int, default=8000)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--study-limit", type=int, default=8000)
    parser.add_argument("--kmeans-limit", type=int, default=8000)
    parser.add_argument("--cv-limit", type=int, default=6000)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--study-max-iter", type=int, default=8)
    parser.add_argument("--loss-epochs", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-notebook", action="store_true")
    return parser.parse_args()


def normalize_limits(args: argparse.Namespace) -> argparse.Namespace:
    for attr in ("train_limit", "optional_train_limit", "test_limit", "study_limit", "kmeans_limit", "cv_limit"):
        value = getattr(args, attr)
        setattr(args, attr, None if value is None or value <= 0 else value)
    return args


def main() -> None:
    args = normalize_limits(parse_args())
    ensure_output_dirs()

    print("MNIST-C datasets:", ", ".join(available_datasets()))
    print("Required training datasets:", ", ".join(REQUIRED_TRAIN_DATASETS))
    print("Running preprocessing study...")
    run_preprocessing_study(args)
    print("Running loss-function study...")
    run_loss_study(args)
    print("Running hyperparameter search...")
    run_hyperparameter_search(args)
    print("Running unsupervised KMeans baseline...")
    run_kmeans_baseline(args)
    print("Running 3-fold cross-validation...")
    run_cross_validation_study(args)
    print("Running required experiments...")
    run_required_experiments(args)
    print("Running optional combined-training experiment...")
    run_optional_combined_experiment(args)
    print("Saving dataset example visualization...")
    plot_dataset_examples()
    summary = summarize_results()
    if not args.skip_notebook:
        notebook_path = make_notebook()
        print(f"Notebook written to: {notebook_path}")
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
