"""Prepare the official xiaoma26 CALVIN LeRobot split for ACT training.

The course staff published environment-specific splits at:
https://huggingface.co/datasets/xiaoma26/calvin-lerobot

This script is the Day 3 production data path. It downloads the full requested
training split, downloads metadata for all official splits, validates the split
metadata, and writes deterministic episode-list views for LeRobot ACT training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse, urlunparse

import requests


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 1024 * 1024
OFFICIAL_EXPECTED = {
    "splitA": {"scene": "A", "total_episodes": 6089, "total_frames": 366693, "total_tasks": 389, "fps": 10},
    "splitB": {"scene": "B", "total_episodes": 6115, "total_frames": 367096, "total_tasks": 389, "fps": 10},
    "splitC": {"scene": "C", "total_episodes": 5666, "total_frames": 337954, "total_tasks": 389, "fps": 10},
    "splitD": {"scene": "D", "total_episodes": 5124, "total_frames": 308918, "total_tasks": 389, "fps": 10},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="xiaoma26/calvin-lerobot")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--download-split", default="splitA")
    parser.add_argument("--metadata-splits", nargs="+", default=["splitA", "splitB", "splitC", "splitD"])
    parser.add_argument(
        "--local-dir",
        default="topic2_act/data/xiaoma26_calvin_lerobot",
    )
    parser.add_argument(
        "--output-dir",
        default="topic2_act/data/splits/xiaoma26_calvin_lerobot",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--smoke-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only split metadata and still write audit/episode views. Used for local dry runs.",
    )
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


def repo_file_name(entry: object, tree_path: str) -> str | None:
    if not isinstance(entry, dict):
        return None

    raw_name = entry.get("path") or entry.get("name")
    entry_type = entry.get("type")
    if not isinstance(raw_name, str) or entry_type == "directory":
        return None

    name = unquote(raw_name).lstrip("/")
    normalized_tree = tree_path.strip("/")
    if normalized_tree and not name.startswith(normalized_tree + "/"):
        name = f"{normalized_tree}/{name}"
    return name


def list_repo_files(
    session: requests.Session,
    *,
    endpoint: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    tree_path: str,
    timeout: float,
) -> list[str]:
    if repo_type != "dataset":
        raise ValueError(f"only repo_type='dataset' is supported, got {repo_type!r}")

    encoded_repo = quote_path(repo_id)
    encoded_revision = quote(revision, safe="")
    encoded_tree_path = quote_path(tree_path.strip("/")) if tree_path else ""
    path_part = f"/{encoded_tree_path}" if encoded_tree_path else ""
    url = (
        f"{endpoint}/api/datasets/{encoded_repo}/tree/{encoded_revision}{path_part}"
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
            name = repo_file_name(entry, tree_path)
            if name:
                files.append(name)

        next_url = extract_next_link(response.headers.get("Link"))
        url = normalize_next_url(next_url, endpoint) if next_url else ""

    return sorted(set(files))


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


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


def download_files(
    session: requests.Session,
    *,
    files: list[str],
    manifest: dict[str, Any],
    manifest_path: Path,
    endpoint: str,
    repo_id: str,
    revision: str,
    local_dir: Path,
    timeout: float,
    retries: int,
    retry_sleep: float,
    force: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, filename in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {filename}", flush=True)
        record = download_one(
            session,
            endpoint=endpoint,
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            local_dir=local_dir,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            force=force,
        )
        print(f"  status: {record['status']} bytes={record.get('bytes', 0)}", flush=True)
        records.append(record)
        manifest["files"].append(record)
        if record["status"] == "failed":
            failures.append(record)
        completed = sum(1 for item in manifest["files"] if item["status"] in {"downloaded", "skipped_existing"})
        manifest["summary"] = {
            "completed": completed,
            "failed": sum(1 for item in manifest["files"] if item["status"] == "failed"),
            "records": len(manifest["files"]),
        }
        write_json_atomic(manifest_path, manifest)

    if failures:
        failed_paths = "\n".join(item["path"] for item in failures[:20])
        raise RuntimeError(f"{len(failures)} downloads failed; first failures:\n{failed_paths}")

    return records


def feature_shape(info: dict[str, Any], key: str) -> list[int] | None:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    return [int(item) for item in shape] if isinstance(shape, list) else None


def validate_split_metadata(local_dir: Path, split: str) -> dict[str, Any]:
    meta_root = local_dir / split / "meta"
    info = read_json(meta_root / "info.json")
    modality = read_json(meta_root / "modality.json")
    episodes = read_jsonl(meta_root / "episodes.jsonl")
    tasks = read_jsonl(meta_root / "tasks.jsonl")
    stats_path = meta_root / "episodes_stats.jsonl"
    stats_rows = read_jsonl(stats_path) if stats_path.exists() else []

    expected = OFFICIAL_EXPECTED.get(split, {})
    failures: list[str] = []
    warnings: list[str] = []
    scene = str(info.get("scene") or "")
    episode_scenes = Counter(str(row.get("scene") or "") for row in episodes)
    episode_indices = [int(row["episode_index"]) for row in episodes]
    total_length = sum(int(row.get("length") or 0) for row in episodes)

    if expected:
        if scene != expected["scene"]:
            failures.append(f"info.scene={scene!r} does not match expected {expected['scene']!r}")
        if int(info.get("total_episodes") or -1) != expected["total_episodes"]:
            failures.append(
                f"info.total_episodes={info.get('total_episodes')} does not match expected {expected['total_episodes']}"
            )
        if int(info.get("total_frames") or -1) != expected["total_frames"]:
            failures.append(
                f"info.total_frames={info.get('total_frames')} does not match expected {expected['total_frames']}"
            )
        if int(info.get("total_tasks") or -1) != expected["total_tasks"]:
            failures.append(f"info.total_tasks={info.get('total_tasks')} does not match expected {expected['total_tasks']}")
        if int(info.get("fps") or -1) != expected["fps"]:
            failures.append(f"info.fps={info.get('fps')} does not match expected {expected['fps']}")

    if len(episodes) != int(info.get("total_episodes") or -1):
        failures.append(f"episodes.jsonl count {len(episodes)} != info.total_episodes {info.get('total_episodes')}")
    if total_length != int(info.get("total_frames") or -1):
        failures.append(f"sum(episode.length) {total_length} != info.total_frames {info.get('total_frames')}")
    if len(tasks) != int(info.get("total_tasks") or -1):
        failures.append(f"tasks.jsonl count {len(tasks)} != info.total_tasks {info.get('total_tasks')}")
    if stats_rows and len(stats_rows) != len(episodes):
        failures.append(f"episodes_stats.jsonl count {len(stats_rows)} != episodes.jsonl count {len(episodes)}")
    if episode_indices != list(range(len(episode_indices))):
        warnings.append("episode_index is not contiguous from 0; inspect before using LeRobot episode filters")
    if len(episode_scenes) != 1 or scene not in episode_scenes:
        failures.append(f"episode scene labels are not uniformly {scene!r}: {dict(episode_scenes)}")

    required_shapes = {
        "state": [15],
        "actions": [7],
        "image": [200, 200, 3],
        "wrist_image": [84, 84, 3],
        "task_index": [1],
    }
    observed_shapes = {key: feature_shape(info, key) for key in required_shapes}
    for key, expected_shape in required_shapes.items():
        if observed_shapes[key] != expected_shape:
            failures.append(f"feature {key!r} shape {observed_shapes[key]} != expected {expected_shape}")

    if "annotation" not in modality or "human.action.task_description" not in modality.get("annotation", {}):
        failures.append("modality.annotation.human.action.task_description is missing")

    return {
        "split": split,
        "scene": scene,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "info": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "total_tasks": info.get("total_tasks"),
            "fps": info.get("fps"),
            "data_path": info.get("data_path"),
            "video_path": info.get("video_path"),
        },
        "episode_count": len(episodes),
        "episode_scene_counts": dict(episode_scenes),
        "episode_length_sum": total_length,
        "task_count": len(tasks),
        "stats_count": len(stats_rows) if stats_rows else None,
        "schema": {
            "state": observed_shapes["state"],
            "actions": observed_shapes["actions"],
            "image": observed_shapes["image"],
            "wrist_image": observed_shapes["wrist_image"],
            "task_index": observed_shapes["task_index"],
            "episode_fields": sorted(episodes[0].keys()) if episodes else [],
        },
        "first_episode": episodes[0] if episodes else None,
    }


def parquet_probe(local_dir: Path, split: str) -> dict[str, Any]:
    parquet = next(iter(sorted((local_dir / split).glob("data/**/*.parquet"))), None)
    if parquet is None:
        return {"available": False, "reason": "no parquet files found"}

    try:
        import pandas as pd
    except ImportError as exc:
        return {"available": False, "path": str(parquet), "reason": f"pandas import failed: {exc}"}

    try:
        frame = pd.read_parquet(parquet)
    except Exception as exc:  # noqa: BLE001 - audit output should preserve exact parquet failure.
        return {"available": False, "path": str(parquet), "reason": f"read_parquet failed: {exc}"}

    columns: list[dict[str, Any]] = []
    for column in frame.columns:
        value = frame[column].iloc[0] if len(frame) else None
        columns.append(
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "sample_type": type(value).__name__,
                "sample_shape": list(getattr(value, "shape", []) or []),
            }
        )
    return {"available": True, "path": str(parquet), "rows": len(frame), "columns": columns}


def split_scene_label(split: str) -> str:
    expected = OFFICIAL_EXPECTED.get(split)
    if expected:
        return str(expected["scene"])
    if split.lower().startswith("split") and len(split) > len("split"):
        return split[len("split") :].upper()
    return split.upper()


def write_episode_views(local_dir: Path, output_dir: Path, split: str, smoke_count: int, seed: int) -> dict[str, Any]:
    episodes = read_jsonl(local_dir / split / "meta" / "episodes.jsonl")
    episode_indices = sorted(int(row["episode_index"]) for row in episodes)
    if smoke_count < 1:
        raise ValueError("--smoke-count must be >= 1")
    if smoke_count > len(episode_indices):
        raise ValueError(f"--smoke-count {smoke_count} exceeds available episodes {len(episode_indices)}")

    shuffled = episode_indices[:]
    random.Random(seed).shuffle(shuffled)
    smoke = sorted(shuffled[:smoke_count])

    output_dir.mkdir(parents=True, exist_ok=True)
    scene = split_scene_label(split)
    full_path = output_dir / f"episodes_{scene}_full.json"
    smoke_path = output_dir / f"episodes_{scene}_smoke{smoke_count}.json"
    write_json_atomic(full_path, episode_indices)
    write_json_atomic(smoke_path, smoke)
    result = {
        "split": split,
        "scene": scene,
        "episodes_full": str(full_path),
        "episodes_full_count": len(episode_indices),
        "episodes_smoke": str(smoke_path),
        "episodes_smoke_count": len(smoke),
        f"episodes_{scene}_full": str(full_path),
        f"episodes_{scene}_full_count": len(episode_indices),
        f"episodes_{scene}_smoke": str(smoke_path),
        f"episodes_{scene}_smoke_count": len(smoke),
        "seed": seed,
    }
    if scene == "A":
        result.update(
            {
                "episodes_A_full": str(full_path),
                "episodes_A_full_count": len(episode_indices),
                "episodes_A_smoke": str(smoke_path),
                "episodes_A_smoke_count": len(smoke),
            }
        )
    return result


def copy_manifest_to_output(manifest_path: Path, output_dir: Path) -> Path:
    output_manifest = output_dir / "download_manifest.json"
    if manifest_path.resolve() != output_manifest.resolve():
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, output_manifest)
    return output_manifest


def main() -> int:
    args = parse_args()
    if args.retries < 1:
        raise ValueError("--retries must be >= 1")
    if args.download_split not in args.metadata_splits:
        args.metadata_splits.append(args.download_split)

    endpoint = args.endpoint.rstrip("/") if args.endpoint else DEFAULT_HF_ENDPOINT
    local_dir = Path(args.local_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "download_manifest.json"
    session = requests.Session()

    manifest: dict[str, Any] = {
        "repo_id": args.repo_id,
        "endpoint": endpoint,
        "revision": args.revision,
        "download_split": args.download_split,
        "metadata_splits": args.metadata_splits,
        "local_dir": str(local_dir),
        "output_dir": str(output_dir),
        "metadata_only": args.metadata_only,
        "started_at": utc_now(),
        "files": [],
        "summary": {},
    }
    write_json_atomic(manifest_path, manifest)

    print(f"repo: {args.repo_id}")
    print(f"endpoint: {endpoint}")
    print(f"revision: {args.revision}")
    print(f"download_split: {args.download_split}")
    print(f"metadata_splits: {args.metadata_splits}")
    print(f"metadata_only: {args.metadata_only}")

    metadata_files: list[str] = []
    for split in args.metadata_splits:
        files = list_repo_files(
            session,
            endpoint=endpoint,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            tree_path=f"{split}/meta",
            timeout=args.timeout,
        )
        if not files:
            raise RuntimeError(f"no metadata files found for {split}")
        print(f"{split}: metadata_file_count={len(files)}")
        metadata_files.extend(files)

    download_files(
        session,
        files=sorted(set(metadata_files)),
        manifest=manifest,
        manifest_path=manifest_path,
        endpoint=endpoint,
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=local_dir,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        force=args.force,
    )

    if not args.metadata_only:
        split_files = list_repo_files(
            session,
            endpoint=endpoint,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            tree_path=args.download_split,
            timeout=args.timeout,
        )
        if not split_files:
            raise RuntimeError(f"no files found for {args.download_split}")
        print(f"{args.download_split}: total_file_count={len(split_files)}")
        download_files(
            session,
            files=split_files,
            manifest=manifest,
            manifest_path=manifest_path,
            endpoint=endpoint,
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=local_dir,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            force=args.force,
        )

    split_summaries = [validate_split_metadata(local_dir, split) for split in args.metadata_splits]
    validation_failed = any(not item["ok"] for item in split_summaries)
    for item in split_summaries:
        counts = item["info"]
        print(
            f"{item['split']}: scene={item['scene']} episodes={counts['total_episodes']} "
            f"frames={counts['total_frames']} tasks={counts['total_tasks']} ok={item['ok']}"
        )
        for warning in item["warnings"]:
            print(f"warning: {item['split']}: {warning}", file=sys.stderr)
        for failure in item["failures"]:
            print(f"error: {item['split']}: {failure}", file=sys.stderr)

    episode_outputs = write_episode_views(local_dir, output_dir, args.download_split, args.smoke_count, args.seed)
    probe = parquet_probe(local_dir, args.download_split)
    output_manifest = copy_manifest_to_output(manifest_path, output_dir)

    summary = {
        "generated_at": utc_now(),
        "repo_id": args.repo_id,
        "endpoint": endpoint,
        "revision": args.revision,
        "download_split": args.download_split,
        "metadata_only": args.metadata_only,
        "local_dir": str(local_dir),
        "output_dir": str(output_dir),
        "official_expected_counts": OFFICIAL_EXPECTED,
        "split_summaries": split_summaries,
        "episode_outputs": episode_outputs,
        "download_manifest": str(output_manifest),
        "parquet_probe": probe,
        "notes": [
            "Official xiaoma26 split is the production path for Day 3 and later A-only training.",
            "Old scene_info.npy reverse splitting is retained only as legacy cross-validation evidence.",
            "splitA/splitB/splitC episode_index values each start at 0; ABC joint training must use multi-dataset loading or remap indices.",
        ],
    }
    summary_path = output_dir / "official_split_summary.json"
    write_json_atomic(summary_path, summary)

    manifest["finished_at"] = utc_now()
    manifest["summary"]["downloaded_bytes"] = sum(
        int(item.get("bytes") or 0) for item in manifest["files"] if item["status"] == "downloaded"
    )
    manifest["summary"]["existing_bytes"] = sum(
        int(item.get("bytes") or 0) for item in manifest["files"] if item["status"] == "skipped_existing"
    )
    manifest["summary"]["summary_path"] = str(summary_path)
    write_json_atomic(manifest_path, manifest)
    copy_manifest_to_output(manifest_path, output_dir)

    if validation_failed:
        print(f"summary: {summary_path}")
        return 1

    print("ok: official CALVIN split preparation completed")
    scene = str(episode_outputs["scene"])
    print(f"episodes_full_count: {episode_outputs['episodes_full_count']}")
    print(f"episodes_smoke_count: {episode_outputs['episodes_smoke_count']}")
    print(f"episodes_{scene}_full_count: {episode_outputs[f'episodes_{scene}_full_count']}")
    print(f"episodes_{scene}_smoke_count: {episode_outputs[f'episodes_{scene}_smoke_count']}")
    print(f"summary: {summary_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
