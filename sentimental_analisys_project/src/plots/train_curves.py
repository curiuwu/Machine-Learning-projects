import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import LearningCurveDisplay

from src.config import PLOTS_DIR

def build_and_save_sklearn_learning_curve(
        model, 
        X, 
        y, 
        output_dir: Path = PLOTS_DIR,
        filename: str = "learning_curve.png"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    fig, ax = plt.subplots(figsize=(7, 5))
    LearningCurveDisplay.from_estimator(
        estimator=model,
        X=X,
        y=y,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        ax=ax
    )

    ax.set_title("Learning Curve")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


    return output_path

def build_and_save_torch_training_curves(
        history: dict[str, list[float]],
        output_dir: Path = PLOTS_DIR,
        filename: str = "training_curves.png",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_f1"], marker="o", label="Train F1 score")
    axes[1].plot(epochs, history["val_f1"], marker="o", label="Validation F1 score")
    axes[1].set_title("F1-macro score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


    return output_path
    