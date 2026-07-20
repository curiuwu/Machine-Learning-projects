import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.build_dataset import load_and_save_data


def main() -> None:
    output_path = load_and_save_data()
    print(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    main()
