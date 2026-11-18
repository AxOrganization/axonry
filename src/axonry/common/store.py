"""Object stores: R2 (S3 API) in production, a local directory for development and tests.

Writes to keys that layout.is_mutable() rejects are write-once: put() refuses to overwrite them with different bytes
(same bytes is a no-op, so retries are idempotent).
"""
from pathlib import Path
from typing import Iterator, Protocol

from . import layout


class ImmutableObjectError(RuntimeError):
    pass


class ObjectStore(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, data: bytes, content_type: str = "application/json") -> None: ...
    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None: ...
    def list(self, prefix: str) -> Iterator[tuple[str, int]]: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def last_modified(self, key: str) -> float | None: ...


def _check_write_once(store, key, data, public):
    if not public or layout.is_mutable(key):
        return True
    old = store.get(key)
    if old is None:
        return True
    if old == data:
        return False
    raise ImmutableObjectError(f"refusing to change immutable object {key}")


class LocalStore:
    def __init__(self, root, public=False):
        self.root, self.public = Path(root).resolve(), public

    def _p(self, key):
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"key escapes store: {key}")
        return p

    def get(self, key):
        p = self._p(key)
        return p.read_bytes() if p.is_file() else None

    def put(self, key, data, content_type="application/json"):
        if not _check_write_once(self, key, data, self.public):
            return
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def put_file(self, key, path, content_type="application/octet-stream"):
        self.put(key, Path(path).read_bytes(), content_type)

    def list(self, prefix):
        base = self._p(prefix) if prefix else self.root
        start = base if base.is_dir() else base.parent
        if not start.exists():
            return
        for p in sorted(start.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                key = p.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    yield key, p.stat().st_size

    def delete_prefix(self, prefix):
        n = 0
        for key, _ in list(self.list(prefix)):
            self._p(key).unlink()
            n += 1
        return n

    def last_modified(self, key):
        p = self._p(key)
        return p.stat().st_mtime if p.is_file() else None


class R2Store:
    """bucket on Cloudflare R2 through the S3 API. Credentials may be a temporary triple (with session token)."""

    def __init__(self, endpoint, bucket, access_key_id, secret_access_key, session_token=None, public=False):
        import boto3  # noqa: PLC0415

        self.bucket, self.public = bucket, public
        self.s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id=access_key_id,
                               aws_secret_access_key=secret_access_key, aws_session_token=session_token)

    def get(self, key):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def _cache(self, key):
        if not self.public:
            return {}
        return {"CacheControl": layout.MUTABLE_CACHE if layout.is_mutable(key) else layout.IMMUTABLE_CACHE}

    def put(self, key, data, content_type="application/json"):
        if not _check_write_once(self, key, data, self.public):
            return
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
                           **self._cache(key))

    def put_file(self, key, path, content_type="application/octet-stream"):
        if self.public and not layout.is_mutable(key) and self.last_modified(key) is not None:
            return  # immutable large object already present (callers verify by manifest sha)
        self.s3.upload_file(str(path), self.bucket, key,
                            ExtraArgs={"ContentType": content_type, **self._cache(key)})

    def list(self, prefix):
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                yield o["Key"], o["Size"]

    def delete_prefix(self, prefix):
        keys, n = [k for k, _ in self.list(prefix)], 0
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            self.s3.delete_objects(Bucket=self.bucket,
                                   Delete={"Objects": [{"Key": k} for k in batch]})
            n += len(batch)
        return n

    def last_modified(self, key):
        try:
            return self.s3.head_object(Bucket=self.bucket, Key=key)["LastModified"].timestamp()
        except Exception:
            return None
