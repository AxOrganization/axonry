"""Operator configuration: one TOML file (config/example.toml documents every key). Secrets never live in the file:
keys ending in `_env` name the environment variable that holds the value (see `secret`)."""
import os
import tomllib
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("AXONRY_CONFIG", "config/axonry.toml"))


class Config:
    def __init__(self, data: dict, path: Path | None = None):
        self.data, self.path = data, path

    @classmethod
    def load(cls, path=None):
        path = Path(path or DEFAULT_PATH)
        with open(path, "rb") as f:
            return cls(tomllib.load(f), path)

    def get(self, dotted: str, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def require(self, dotted: str):
        v = self.get(dotted)
        if v is None:
            raise KeyError(f"missing config key {dotted} in {self.path}")
        return v

    def secret(self, dotted: str) -> str:
        """Value of the environment variable named by `<dotted>_env`."""
        var = self.require(dotted + "_env")
        v = os.environ.get(var)
        if not v:
            raise KeyError(f"environment variable {var} (config {dotted}_env) is not set")
        return v

    def path_of(self, dotted: str) -> Path:
        p = Path(self.require(dotted)).expanduser()
        return p if p.is_absolute() or self.path is None else (self.path.parent.parent / p)


def public_store(cfg: Config):
    if cfg.get("storage.backend", "r2") == "local":
        from .store import LocalStore  # noqa: PLC0415
        return LocalStore(cfg.path_of("storage.local_public"), public=True)
    from .store import R2Store  # noqa: PLC0415
    return R2Store(cfg.require("r2.endpoint"), cfg.require("r2.public_bucket"), cfg.secret("r2.access_key_id"),
                   cfg.secret("r2.secret_access_key"), public=True)


def private_store(cfg: Config):
    """The private bucket."""
    if cfg.get("storage.backend", "r2") == "local":
        from .store import LocalStore  # noqa: PLC0415
        return LocalStore(cfg.path_of("storage.local_private"))
    from .store import R2Store  # noqa: PLC0415
    return R2Store(cfg.require("r2.endpoint"), cfg.require("r2.private_bucket"), cfg.secret("r2.access_key_id"),
                   cfg.secret("r2.secret_access_key"))
