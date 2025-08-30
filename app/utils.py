from pathlib import Path

def resolve_path(relative_path: str) -> Path:

    try:
        base_dir = Path(__file__).resolve().parent.parent
    except NameError:
        base_dir = Path().resolve()
    return base_dir / relative_path