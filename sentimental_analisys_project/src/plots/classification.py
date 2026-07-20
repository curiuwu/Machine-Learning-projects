from pathlib import Path
import json

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.config import PLOTS_DIR, REPORTS_DIR

def build_and_save_confusion_matrix(
        y_true,
        y_pred,
        labels: list[int],
        class_names: list[str],
        output_dir: Path = PLOTS_DIR,
        filename: str = "confusion_matrix.png"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        data=cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path

def make_classification_report(
        y_true,
        y_pred,
        labels: list[int],
        class_names: list[str],
        output_dir: Path = REPORTS_DIR,
        filename: str = "classification_report.json"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    with output_path.open("w", encoding="utf8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return output_path

def make_classification_report_txt(
        y_true,
        y_pred,
        labels: list[int],
        class_names: list[str],
        output_dir: Path = REPORTS_DIR,
        filename: str = "classification_report.txt"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )

    output_path.write_text(report, encoding="utf8")


    return output_path
