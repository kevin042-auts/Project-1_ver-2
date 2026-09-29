from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import pandas as pd


UploadedFileLike = object


def fingerprint_uploaded_files(uploaded_files: object) -> str:
    digest = hashlib.sha256()
    for uploaded_file in normalize_uploaded_files(uploaded_files):
        name = str(getattr(uploaded_file, "name", "")).encode("utf-8")
        content = getattr(uploaded_file, "getvalue", lambda: b"")()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def normalize_uploaded_files(uploaded_files: Union[UploadedFileLike, Sequence[UploadedFileLike], None]) -> List[UploadedFileLike]:
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (list, tuple)):
        return list(uploaded_files)
    return [uploaded_files]


def read_uploaded_ledger(uploaded_file: UploadedFileLike) -> pd.DataFrame:
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower()
    raw_bytes = getattr(uploaded_file, "getvalue", lambda: b"")()
    if not raw_bytes:
        raise ValueError(f"No content found in uploaded file: {getattr(uploaded_file, 'name', 'unknown')}")

    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(raw_bytes))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw_bytes))

    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def combine_uploaded_files(uploaded_files: Union[UploadedFileLike, Sequence[UploadedFileLike], None]) -> pd.DataFrame:
    normalized = normalize_uploaded_files(uploaded_files)
    frames = [read_uploaded_ledger(file) for file in normalized]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
