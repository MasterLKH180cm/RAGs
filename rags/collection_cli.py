"""Collection-oriented commands using the authenticated public client."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from rags.client import ClientError, HttpClient, MAX_BYTES
from rags.file_io import BoundedFileError, read_regular_bytes
from rags.model_profiles import strict_json


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--url", help="RAG HTTP endpoint (defaults to RAGS_URL or loopback)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--allow-insecure-http", action="store_true",
                        help="Explicitly permit plaintext HTTP outside loopback")


def add_commands(subparsers) -> None:
    collection = subparsers.add_parser("collection", help="Manage tenant-scoped collection datasets")
    actions = collection.add_subparsers(dest="collection_command", required=True)
    create = actions.add_parser("create", help="Create a collection with a fixed embedding configuration")
    create.add_argument("name")
    create.add_argument("--metadata-file", type=Path)
    create.add_argument("--embedding-file", type=Path)
    create.add_argument("--sparse-space-id")
    create.add_argument("--get-or-create", action="store_true")
    listing = actions.add_parser("list", help="List a bounded page of visible collections")
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--offset", type=int, default=0)
    collection_parsers = [create, listing]
    for action in ("get", "count", "status", "modify", "delete", "fork"):
        command = actions.add_parser(action)
        command.add_argument("collection")
        if action in {"modify", "fork"}:
            command.add_argument("--name", required=action == "fork")
            command.add_argument("--metadata-file", type=Path)
        if action in {"modify", "delete", "fork"}:
            command.add_argument("--expected-generation", type=int)
        collection_parsers.append(command)
    for parser in collection_parsers:
        _common(parser)

    records = subparsers.add_parser("records", help="Write, inspect or delete collection records")
    record_actions = records.add_subparsers(dest="records_command", required=True)
    for action in ("add", "update", "upsert", "get", "delete"):
        command = record_actions.add_parser(action)
        command.add_argument("collection")
        if action in {"add", "update", "upsert"}:
            command.add_argument("--file", type=Path, required=True,
                                 help="JSON object containing ids and aligned data arrays")
        else:
            selection = command.add_mutually_exclusive_group(required=action == "delete")
            selection.add_argument("--file", type=Path, help="Complete record selection JSON object")
            selection.add_argument("--ids", nargs="+")
            if action == "get":
                command.add_argument("--limit", type=int)
                command.add_argument("--offset", type=int)
            else:
                command.add_argument("--expected-generation", type=int)
        _common(command)

    search = subparsers.add_parser("search", help="Search a collection using dense, BM25, sparse or hybrid ranking")
    search.add_argument("collection")
    query = search.add_mutually_exclusive_group(required=True)
    query.add_argument("--query-file", type=Path, help="UTF-8 query text; at most 2000 characters")
    query.add_argument("--file", type=Path, help="Complete JSON object containing searches")
    search.add_argument("--mode", choices=("dense", "lexical", "sparse", "hybrid"))
    search.add_argument("--limit", type=int)
    search.add_argument("--offset", type=int)
    search.add_argument("--where-file", type=Path)
    search.add_argument("--where-document-file", type=Path)
    search.add_argument("--select", nargs="+", choices=("id", "version", "document", "metadata",
                                                       "embedding", "sparse_embedding", "score", "distance"))
    _common(search)


def _read_file(path: Path, *, limit: int = MAX_BYTES) -> bytes:
    """Bound reads before allocation and avoid following a swapped symlink."""
    try:
        return read_regular_bytes(path, max_bytes=limit)
    except BoundedFileError as exc:
        if exc.reason == "read_limit_exceeded":
            raise ClientError("request_too_large", 413) from None
        raise ClientError("invalid_input_file", 422) from None
    except OSError:
        raise ClientError("invalid_input_file", 422) from None


def _read_json(path: Path) -> dict:
    try:
        data = strict_json(_read_file(path))
        if not isinstance(data, dict):
            raise ValueError
        return data
    except ClientError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise ClientError("invalid_input_file", 422) from None


def _collection_command(client: HttpClient, args) -> dict:
    action = args.collection_command
    if action == "list":
        return client.list_collections_page(limit=args.limit, offset=args.offset)
    if action == "create":
        options = {}
        if args.metadata_file is not None:
            options["metadata"] = _read_json(args.metadata_file)
        if args.embedding_file is not None:
            options["embedding"] = _read_json(args.embedding_file)
        if args.sparse_space_id is not None:
            options["sparse_space_id"] = args.sparse_space_id
        method = client.get_or_create_collection if args.get_or_create else client.create_collection
        return method(args.name, **options).to_dict()
    collection = client.get_collection(args.collection)
    if action == "get":
        return collection.to_dict()
    if action == "count":
        return {"collection_id": collection.id, "count": collection.count()}
    if action == "status":
        return collection.indexing_status()
    options = {}
    if args.expected_generation is not None:
        options["expected_generation"] = args.expected_generation
    if action == "delete":
        return client.delete_collection(collection_id=collection.id, **options)
    if args.metadata_file is not None:
        options["metadata"] = _read_json(args.metadata_file)
    if action == "fork":
        return collection.fork(args.name, **options).to_dict()
    if args.name is not None:
        options["name"] = args.name
    return collection.modify(**options)


def _records_command(client: HttpClient, args) -> dict:
    action = args.records_command
    data = _read_json(args.file) if args.file is not None else {}
    if action in {"get", "delete"}:
        for name in ("ids", "limit", "offset", "expected_generation"):
            value = getattr(args, name, None)
            if value is not None:
                if name in data:
                    raise ClientError("ambiguous_input", 422)
                data[name] = value
    collection = client.get_collection(args.collection)
    return getattr(collection, action)(**data)


def _search_command(client: HttpClient, args) -> dict:
    if args.file is not None:
        if any(getattr(args, name) is not None for name in (
            "mode", "limit", "offset", "where_file", "where_document_file", "select"
        )):
            raise ClientError("ambiguous_input", 422)
        data = _read_json(args.file)
    else:
        try:
            text = _read_file(args.query_file, limit=8000).decode("utf-8").strip()
        except UnicodeError:
            raise ClientError("invalid_input_file", 422) from None
        if not 1 <= len(text) <= 2000:
            raise ClientError("invalid_input_file", 422)
        query = {"query_text": text}
        for key in ("mode", "limit", "offset", "select"):
            value = getattr(args, key)
            if value is not None:
                query[key] = value
        for key, path in (("where", args.where_file), ("where_document", args.where_document_file)):
            if path is not None:
                query[key] = _read_json(path)
        data = {"searches": [query]}
    collection = client.get_collection(args.collection)
    return collection.search(**data)


def run(args) -> int:
    """Environment loading remains owned by the existing CLI entry point."""
    try:
        with HttpClient(args.url or os.getenv("RAGS_URL", "http://127.0.0.1:8000"),
                        os.getenv("RAGS_API_KEY", ""), timeout=args.timeout,
                        allow_insecure_http=args.allow_insecure_http) as client:
            if args.command == "collection":
                result = _collection_command(client, args)
            elif args.command == "records":
                result = _records_command(client, args)
            else:
                result = _search_command(client, args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except ClientError as exc:
        print(json.dumps({"error": exc.code, "status_code": exc.status_code}), file=sys.stderr)
        return 1
    except (ValueError, TypeError, OSError):
        print(json.dumps({"error": "invalid_input", "status_code": 422}), file=sys.stderr)
        return 1
