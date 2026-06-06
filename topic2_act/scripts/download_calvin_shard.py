"""Download one CALVIN LeRobot shard from a Hugging Face-compatible mirror.

This script intentionally uses the same manual mirror API strategy as the Day 1
probe script. The hosted server is in a restricted network environment, and the
standard huggingface_hub pagination may jump back to huggingface.co.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse, urlunparse

import requests


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="huiwon/calvin_task_ABC_D")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prefix", default="calvin_task_ABC_D_lerobot_0_4/")
    parser.add_argument(
        "--local-dir",
        default="/root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_first_shard",
    )
    parser.add_argument(
        "--manifest",
        default="/root/Test/Zhr/DL/HW3/topic2_act/data/calvin_task_ABC_D_first_shard_manifest.json",
    )
    parser.add_argument("--failures", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT,
        help="Hugging Face endpoint. Defaults to HF_ENDPOINT or hf-mirror.com.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise ValueError(f"only repo_type='dataset' is supported, got {repo_type!r}")

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

        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected tree response type: url={url}, type={type(payload).__name__}")

        for entry in payload:
            name = repo_file_name(entry)
            if name:
                files.append(name)

        next_url = extract_next_link(response.headers.get("Link"))
        url = normalize_next_url(next_url, endpoint) if next_url else ""

    return files


def resolve_url(endpoint: str, repo_id: str, revision: str, filename: str) -> str:
    encoded_repo = quote_path(repo_id)
    encoded_revision = quote(revision, safe="")
    encoded_filename = quote_path(filename)
    return f"{endpoint}/datasets/{encoded_repo}/resolve/{encoded_revision}/{encoded_filename}"


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)


def default_failures_path(manifest: Path) -> Path:
    return manifest.with_name(f"{manifest.stem}_failures.json")


def download_one(
    session: requests.Session,
    *,
    endpoint: str,
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: Path,
    timeout: float,
    retries: int,
    retry_sleep: float,
    force: bool,
) -> dict[str, Any]:
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0 and not force:
        return {
            "path": filename,
            "local_path": str(target),
            "status": "skipped_existing",
            "bytes": target.stat().st_size,
            "completed_at": utc_now(),
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    url = resolve_url(endpoint, repo_id, revision, filename)
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            bytes_written = 0
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with part.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            file.write(chunk)
                            bytes_written += len(chunk)
            part.replace(target)
            return {
                "path": filename,
                "local_path": str(target),
                "status": "downloaded",
                "bytes": bytes_written,
                "attempts": attempt,
                "completed_at": utc_now(),
            }
        except requests.RequestException as exc:
            status = getattr(exc.response, "status_code", None)
            last_error = f"status={status}, error={type(exc).__name__}: {exc}"
        except OSError as exc:
            last_error = f"os_error={type(exc).__name__}: {exc}"

        if attempt < retries:
            print(f"warning: retrying {filename} after attempt {attempt}: {last_error}", flush=True)
            time.sleep(retry_sleep)

    return {
        "path": filename,
        "local_path": str(target),
        "status": "failed",
        "error": last_error,
        "completed_at": utc_now(),
    }


def main() -> int:
    args = parse_args()
    endpoint = args.endpoint.rstrip("/") if args.endpoint else DEFAULT_HF_ENDPOINT
    local_dir = Path(args.local_dir)
    manifest_path = Path(args.manifest)
    failures_path = Path(args.failures) if args.failures else default_failures_path(manifest_path)
    prefix = args.prefix

    if args.retries < 1:
        raise ValueError("--retries must be >= 1")

    session = requests.Session()
    print(f"repo: {args.repo_id}")
    print(f"endpoint: {endpoint}")
    print(f"revision: {args.revision}")
    print(f"prefix: {prefix}")
    print(f"local_dir: {local_dir}")

    files = list_repo_files(
        session,
        endpoint=endpoint,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        timeout=args.timeout,
    )
    selected = sorted(name for name in files if name.startswith(prefix))
    print(f"repo_file_count: {len(files)}")
    print(f"selected_file_count: {len(selected)}")

    if not selected:
        raise RuntimeError(f"no files matched prefix {prefix!r}")

    manifest: dict[str, Any] = {
        "repo_id": args.repo_id,
        "endpoint": endpoint,
        "revision": args.revision,
        "prefix": prefix,
        "local_dir": str(local_dir),
        "started_at": utc_now(),
        "dry_run": args.dry_run,
        "total_files": len(selected),
        "files": [],
        "summary": {},
    }

    if args.dry_run:
        manifest["files"] = [{"path": name, "status": "planned"} for name in selected]
        manifest["summary"] = {"planned": len(selected)}
        manifest["finished_at"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        print(f"dry_run_manifest: {manifest_path}")
        return 0

    failures: list[dict[str, Any]] = []
    for idx, filename in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {filename}", flush=True)
        record = download_one(
            session,
            endpoint=endpoint,
            repo_id=args.repo_id,
            revision=args.revision,
            filename=filename,
            local_dir=local_dir,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            force=args.force,
        )
        print(f"  status: {record['status']} bytes={record.get('bytes', 0)}", flush=True)
        manifest["files"].append(record)
        if record["status"] == "failed":
            failures.append(record)

        completed = sum(1 for item in manifest["files"] if item["status"] in {"downloaded", "skipped_existing"})
        manifest["summary"] = {
            "completed": completed,
            "failed": len(failures),
            "remaining": len(selected) - len(manifest["files"]),
        }
        write_json_atomic(manifest_path, manifest)
        if failures:
            write_json_atomic(failures_path, failures)

    manifest["finished_at"] = utc_now()
    manifest["summary"]["downloaded_bytes"] = sum(
        int(item.get("bytes") or 0) for item in manifest["files"] if item["status"] == "downloaded"
    )
    manifest["summary"]["existing_bytes"] = sum(
        int(item.get("bytes") or 0) for item in manifest["files"] if item["status"] == "skipped_existing"
    )
    write_json_atomic(manifest_path, manifest)

    if failures:
        write_json_atomic(failures_path, failures)
        print(f"failed_files: {len(failures)}")
        print(f"failures: {failures_path}")
        return 1

    if failures_path.exists():
        failures_path.unlink()
    print(f"ok: downloaded or found existing files for prefix {prefix}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
