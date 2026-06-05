"""Probe the course CALVIN LeRobot dataset without downloading everything.

The Day 1 goal is to confirm the repository structure and collect enough schema
evidence for the Day 2 data audit. By default this script downloads metadata
files and one parquet episode only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, unquote, urlparse, urlunparse

import pandas as pd
import requests


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="huiwon/calvin_task_ABC_D")
    parser.add_argument(
        "--local-dir",
        default="/root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_probe",
    )
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--max-meta-files", type=int, default=50)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT,
        help="Hugging Face endpoint. Defaults to HF_ENDPOINT or hf-mirror.com.",
    )
    return parser.parse_args()


def first_or_none(items: Iterable[str]) -> str | None:
    for item in items:
        return item
    return None


def quote_path(path: str) -> str:
    return "/".join(quote(part, safe="") for part in path.split("/"))


def endpoint_parts(endpoint: str) -> tuple[str, str]:
    parsed = urlparse(endpoint.rstrip("/"))
    return parsed.scheme, parsed.netloc


def normalize_next_url(next_url: str, endpoint: str) -> str:
    scheme, netloc = endpoint_parts(endpoint)
    parsed = urlparse(next_url)
    return urlunparse(parsed._replace(scheme=scheme, netloc=netloc))


def extract_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None

    for item in link_header.split(","):
        section = item.strip()
        if 'rel="next"' not in section and "rel=next" not in section:
            continue
        start = section.find("<")
        end = section.find(">", start + 1)
        if start != -1 and end != -1:
            return section[start + 1 : end]

    return None


def repo_file_name(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None

    raw_name = entry.get("path") or entry.get("name")
    entry_type = entry.get("type")
    if not isinstance(raw_name, str) or entry_type == "directory":
        return None
    return unquote(raw_name)


def list_repo_files(
    session: requests.Session,
    *,
    endpoint: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    timeout: float,
) -> list[str]:
    if repo_type != "dataset":
        raise ValueError(f"manual mirror probe only supports repo_type='dataset', got {repo_type!r}")

    encoded_repo = quote_path(repo_id)
    encoded_revision = quote(revision, safe="")
    url = (
        f"{endpoint}/api/datasets/{encoded_repo}/tree/{encoded_revision}"
        "?recursive=true&expand=false&limit=1000"
    )
    files: list[str] = []
    seen_urls: set[str] = set()

    while url:
        if url in seen_urls:
            raise RuntimeError(f"pagination loop detected at {url}")
        seen_urls.add(url)

        response = session.get(url, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(exc.response, "status_code", response.status_code)
            raise RuntimeError(
                f"request failed: url={url}, status={status}, error={type(exc).__name__}: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"invalid JSON response: url={url}, error={exc}") from exc

        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected tree response type: url={url}, type={type(payload).__name__}")

        for entry in payload:
            name = repo_file_name(entry)
            if name:
                files.append(name)

        next_url = extract_next_link(response.headers.get("Link"))
        url = normalize_next_url(next_url, endpoint) if next_url else ""

    return files


def download_file(
    session: requests.Session,
    *,
    endpoint: str,
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: Path,
    timeout: float,
) -> Path:
    encoded_repo = quote_path(repo_id)
    encoded_revision = quote(revision, safe="")
    encoded_filename = quote_path(filename)
    url = f"{endpoint}/datasets/{encoded_repo}/resolve/{encoded_revision}/{encoded_filename}"
    target = local_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with target.open("wb") as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        raise RuntimeError(
            f"download failed: url={url}, status={status}, error={type(exc).__name__}: {exc}"
        ) from exc

    return target


def maybe_print_json(path: Path) -> None:
    if path.suffix not in {".json", ".jsonl"}:
        return

    print(f"\n--- preview: {path.name} ---")
    with path.open("r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if idx >= 5:
                break
            line = line.strip()
            if not line:
                continue
            try:
                print(json.dumps(json.loads(line), ensure_ascii=False)[:1000])
            except json.JSONDecodeError:
                print(line[:1000])


def print_parquet_schema(path: Path) -> None:
    print(f"\n--- parquet sample: {path} ---")
    frame = pd.read_parquet(path)
    print(f"rows: {len(frame)}")
    print("columns:")
    for column in frame.columns:
        value = frame[column].iloc[0] if len(frame) else None
        shape = getattr(value, "shape", None)
        print(
            f"- {column}: dtype={frame[column].dtype}, "
            f"sample_type={type(value).__name__}, shape={shape}"
        )


def main() -> int:
    args = parse_args()
    endpoint = args.endpoint.rstrip("/") if args.endpoint else DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = endpoint

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    print(f"repo: {args.repo_id}")
    print(f"endpoint: {endpoint}")
    print(f"revision: {args.revision}")

    files = list_repo_files(
        session,
        endpoint=endpoint,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        timeout=args.timeout,
    )
    print(f"file_count: {len(files)}")

    top_level = sorted({name.split("/")[0] for name in files if "/" in name})
    print("top_level_dirs:")
    for dirname in top_level:
        print(f"- {dirname}")

    meta_files = [name for name in files if "/meta/" in name or name.startswith("meta/")]
    parquet_files = [name for name in files if name.endswith(".parquet")]
    print(f"meta_file_count: {len(meta_files)}")
    print(f"parquet_file_count: {len(parquet_files)}")

    selected_meta = meta_files[: args.max_meta_files]
    selected_parquet = first_or_none(parquet_files)
    selected = selected_meta + ([selected_parquet] if selected_parquet else [])

    downloaded: list[Path] = []
    for filename in selected:
        path = download_file(
            session,
            endpoint=endpoint,
            repo_id=args.repo_id,
            revision=args.revision,
            filename=filename,
            local_dir=local_dir,
            timeout=args.timeout,
        )
        downloaded.append(path)
        print(f"downloaded: {filename}")

    for path in downloaded:
        maybe_print_json(path)

    if selected_parquet is not None:
        sample_path = local_dir / selected_parquet
        print_parquet_schema(sample_path)
    else:
        print("warning: no parquet files found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
