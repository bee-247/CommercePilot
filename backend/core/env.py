"""Project-wide environment loading.

All backend entry points read the same repository-root ``.env`` file so
behavior does not depend on the shell's current working directory.
"""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    load_dotenv(ENV_FILE)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured relative path against the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_sqlite_url(url: str) -> str:
    """Resolve relative SQLite URLs against the repository root."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw_path = url[len(prefix):]
    path = Path(raw_path)
    if path.is_absolute():
        return url
    return f"{prefix}{resolve_project_path(path).resolve()}"
