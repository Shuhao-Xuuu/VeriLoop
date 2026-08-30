"""Guarded, deterministic UTF-8 file tools for one local workspace."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from difflib import unified_diff
from fnmatch import fnmatchcase
import hashlib
import os
from pathlib import Path, PureWindowsPath
import stat
import tempfile
from typing import Any, Iterator

from .protocol import ErrorKind
from .tools import ToolExecutionError


DEFAULT_MAX_FILE_BYTES = 1024 * 1024
MAX_READ_LINES = 500
MAX_LIST_DEPTH = 20
MAX_LIST_RESULTS = 1000
MAX_SEARCH_RESULTS = 500
SEARCH_PREVIEW_CHARS = 300
DIFF_PREVIEW_CHARS = 4000

SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".veriloop",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
    }
)

_SENSITIVE_EXACT_NAMES = frozenset(
    {
        ".env",
        "id_rsa",
        "id_ed25519",
        "credentials.json",
        "serviceaccountkey.json",
    }
)
_PROTECTED_METADATA_COMPONENTS = frozenset({".git", ".veriloop"})


@dataclass(frozen=True, slots=True)
class _ResolvedPath:
    lexical: Path
    resolved: Path
    relative: Path


class WorkspaceGuard:
    """Canonicalize model paths and enforce the file-tool trust boundary."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise ValueError(f"workspace root does not exist: {root_path}")
        if not root_path.is_dir():
            raise ValueError(f"workspace root is not a directory: {root_path}")
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        self._root = root_path.resolve(strict=True)
        self._max_file_bytes = max_file_bytes

    @property
    def root(self) -> Path:
        return self._root

    @property
    def max_file_bytes(self) -> int:
        return self._max_file_bytes

    def resolve(
        self,
        model_path: str,
        *,
        allow_root: bool = False,
    ) -> Path:
        """Resolve a relative model path and prove canonical containment."""

        return self._resolve(model_path, allow_root=allow_root).resolved

    def resolve_for_read(
        self,
        model_path: str,
        *,
        allow_root: bool = False,
    ) -> Path:
        resolved = self._resolve(model_path, allow_root=allow_root)
        if _has_protected_metadata_component(
            Path(model_path).parts,
            PureWindowsPath(model_path).parts,
            resolved.relative.parts,
        ):
            raise ToolExecutionError(
                ErrorKind.PATH_READ_DENIED,
                f"reading protected path is denied: {model_path}",
            )
        if self._is_sensitive(resolved.relative) or self._is_sensitive_lexical(
            model_path
        ):
            raise ToolExecutionError(
                ErrorKind.PATH_READ_DENIED,
                f"reading protected path is denied: {model_path}",
            )
        return resolved.resolved

    def resolve_for_write(self, model_path: str) -> Path:
        resolved = self._resolve(model_path, allow_root=False)
        if _is_link_like(resolved.lexical):
            raise ToolExecutionError(
                ErrorKind.PATH_IS_SYMLINK,
                f"writing through a symlink is denied: {model_path}",
            )
        if self._is_sensitive(resolved.relative) or self._is_sensitive_lexical(
            model_path
        ):
            raise ToolExecutionError(
                ErrorKind.PATH_WRITE_DENIED,
                f"writing protected path is denied: {model_path}",
            )
        if _has_protected_metadata_component(
            Path(model_path).parts,
            PureWindowsPath(model_path).parts,
            resolved.relative.parts,
        ):
            raise ToolExecutionError(
                ErrorKind.PATH_WRITE_DENIED,
                f"writing protected path is denied: {model_path}",
            )
        return resolved.resolved

    def relative(self, path: Path) -> str:
        try:
            relative = path.resolve(strict=False).relative_to(self._root)
        except ValueError as exc:
            raise ToolExecutionError(
                ErrorKind.PATH_OUTSIDE_WORKSPACE,
                "path is outside the workspace",
            ) from exc
        value = relative.as_posix()
        return value if value else "."

    def is_sensitive(self, path: Path) -> bool:
        return self._is_sensitive(path.resolve(strict=False).relative_to(self._root))

    def path_uses_link(self, model_path: str) -> bool:
        """Report any existing link/reparse component in a contained model path."""

        self._resolve(model_path, allow_root=True)
        current = self._root
        for part in Path(model_path).parts:
            if part in {"", "."}:
                continue
            current = current / part
            if _is_link_like(current):
                return True
        return False

    def path_enters_skipped_directory(self, model_path: str) -> bool:
        """Check lexical and canonical components against traversal exclusions."""

        resolved = self._resolve(model_path, allow_root=True)
        for parts in (Path(model_path).parts, resolved.relative.parts):
            current = self._root
            for part in parts:
                if part in {"", "."}:
                    continue
                current = current / part
                if (
                    _policy_component(part) in SKIPPED_DIRECTORIES
                    and (_is_link_like(current) or current.is_dir())
                ):
                    return True
        return False

    def lexical_path(self, model_path: str) -> Path:
        """Return the already-contained lexical path without following its final link."""

        self._resolve(model_path, allow_root=True)
        return self._root / Path(model_path)

    def relative_lexical(self, path: Path) -> str:
        try:
            relative = path.relative_to(self._root)
        except ValueError as exc:
            raise ToolExecutionError(
                ErrorKind.PATH_OUTSIDE_WORKSPACE,
                "path is outside the workspace",
            ) from exc
        value = relative.as_posix()
        return value if value else "."

    def _resolve(self, model_path: str, *, allow_root: bool) -> _ResolvedPath:
        if not isinstance(model_path, str):
            raise ToolExecutionError(
                ErrorKind.INVALID_ARGUMENTS,
                "path must be a string",
            )
        if model_path == "":
            raise ToolExecutionError(
                ErrorKind.INVALID_ARGUMENTS,
                "path must not be empty",
            )

        supplied = Path(model_path)
        windows_path = PureWindowsPath(model_path)
        if (
            supplied.is_absolute()
            or supplied.anchor
            or supplied.drive
            or windows_path.anchor
            or windows_path.drive
        ):
            raise ToolExecutionError(
                ErrorKind.PATH_OUTSIDE_WORKSPACE,
                f"absolute paths are denied: {model_path}",
            )
        if any(":" in part for part in windows_path.parts):
            raise ToolExecutionError(
                ErrorKind.INVALID_ARGUMENTS,
                f"colon path components are denied: {model_path}",
            )
        if ".." in supplied.parts or ".." in windows_path.parts:
            raise ToolExecutionError(
                ErrorKind.PATH_OUTSIDE_WORKSPACE,
                f"parent traversal is denied: {model_path}",
            )

        lexical = self._root / supplied
        try:
            resolved = lexical.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ToolExecutionError(
                ErrorKind.PATH_OUTSIDE_WORKSPACE,
                f"path cannot be resolved safely: {model_path}",
            ) from exc

        if not resolved.is_relative_to(self._root):
            raise ToolExecutionError(
                ErrorKind.PATH_OUTSIDE_WORKSPACE,
                f"path escapes the workspace: {model_path}",
            )
        relative = resolved.relative_to(self._root)
        if not allow_root and resolved == self._root:
            raise ToolExecutionError(
                ErrorKind.INVALID_ARGUMENTS,
                "a file path must not resolve to the workspace root",
            )
        return _ResolvedPath(lexical=lexical, resolved=resolved, relative=relative)

    @staticmethod
    def _is_sensitive(relative: Path) -> bool:
        if not relative.parts:
            return False
        name = _policy_component(relative.name)
        return (
            name in _SENSITIVE_EXACT_NAMES
            or fnmatchcase(name, ".env.*")
            or fnmatchcase(name, "*.pem")
            or fnmatchcase(name, "*.key")
        )

    def _is_sensitive_lexical(self, model_path: str) -> bool:
        supplied = Path(model_path)
        normalized = (self._root / supplied).absolute()
        try:
            relative = normalized.relative_to(self._root)
        except ValueError:
            return False
        return self._is_sensitive(relative)


def list_files(
    guard: WorkspaceGuard,
    *,
    path: str = ".",
    max_depth: int = 3,
    max_results: int = 300,
) -> dict[str, Any]:
    _require_range("max_depth", max_depth, 1, MAX_LIST_DEPTH)
    _require_range("max_results", max_results, 1, MAX_LIST_RESULTS)
    start = guard.resolve_for_read(path, allow_root=True)
    _require_existing(start, path)

    relative_start = guard.relative(start)
    if guard.path_enters_skipped_directory(path):
        return {"path": relative_start, "entries": [], "truncated": False}

    lexical_start = guard.lexical_path(path)
    if _is_link_like(lexical_start):
        return {
            "path": guard.relative_lexical(lexical_start),
            "entries": [
                {"path": guard.relative_lexical(lexical_start), "type": "symlink"}
            ],
            "truncated": False,
        }
    if guard.path_uses_link(path):
        raise ToolExecutionError(
            ErrorKind.PATH_READ_DENIED,
            f"listing through a symlink directory is denied: {path}",
        )
    if not start.is_dir() and not start.is_file():
        raise ToolExecutionError(
            ErrorKind.PATH_READ_DENIED,
            f"path is not a regular file or directory: {path}",
        )

    entries: list[dict[str, Any]] = []
    truncated = False

    def add(item: dict[str, Any]) -> bool:
        nonlocal truncated
        if len(entries) >= max_results:
            truncated = True
            return False
        entries.append(item)
        return True

    if start.is_file():
        add(_list_entry(guard, start))
    else:
        for child, depth in _walk_directory(start, max_depth=max_depth):
            if _is_link_like(child):
                item = {
                    "path": guard.relative_lexical(child),
                    "type": "symlink",
                }
            elif child.is_dir():
                item = {
                    "path": guard.relative_lexical(child),
                    "type": "directory",
                }
            elif child.is_file():
                if guard.is_sensitive(child):
                    continue
                item = _list_entry(guard, child)
            else:
                continue
            if not add(item):
                break

    return {
        "path": guard.relative(start),
        "entries": entries,
        "truncated": truncated,
    }


def read_file(
    guard: WorkspaceGuard,
    *,
    path: str,
    start_line: int = 1,
    end_line: int = 400,
) -> dict[str, Any]:
    if start_line < 1:
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            "start_line must be at least 1",
        )
    if end_line < start_line:
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            "end_line must be greater than or equal to start_line",
        )
    if end_line - start_line + 1 > MAX_READ_LINES:
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            f"a read may return at most {MAX_READ_LINES} lines",
        )

    target = guard.resolve_for_read(path)
    data, text = _read_text_file(guard, target, path)
    lines = text.splitlines()
    selected = lines[start_line - 1 : end_line]
    actual_start = start_line if selected else None
    actual_end = start_line + len(selected) - 1 if selected else None
    numbered = "\n".join(
        f"{number:6d} | {line}"
        for number, line in enumerate(selected, start=start_line)
    )
    return {
        "path": guard.relative(target),
        "requested_range": {"start_line": start_line, "end_line": end_line},
        "actual_range": {"start_line": actual_start, "end_line": actual_end},
        "total_lines": len(lines),
        "sha256": _sha256(data),
        "content": numbered,
        "truncated": bool(lines) and (start_line > 1 or end_line < len(lines)),
    }


def search_text(
    guard: WorkspaceGuard,
    *,
    query: str,
    path: str = ".",
    case_sensitive: bool = False,
    max_results: int = 50,
) -> dict[str, Any]:
    if query == "":
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            "query must not be empty",
        )
    _require_range("max_results", max_results, 1, MAX_SEARCH_RESULTS)
    start = guard.resolve_for_read(path, allow_root=True)
    _require_existing(start, path)

    relative_start = guard.relative(start)
    if guard.path_enters_skipped_directory(path):
        return {
            "query": query,
            "path": relative_start,
            "case_sensitive": case_sensitive,
            "matches": [],
            "truncated": False,
        }

    if _is_link_like(guard.lexical_path(path)):
        return {
            "query": query,
            "path": Path(path).as_posix(),
            "case_sensitive": case_sensitive,
            "matches": [],
            "truncated": False,
        }
    if guard.path_uses_link(path):
        raise ToolExecutionError(
            ErrorKind.PATH_READ_DENIED,
            f"searching through a symlink directory is denied: {path}",
        )

    matches: list[dict[str, Any]] = []
    truncated = False
    needle = query if case_sensitive else query.casefold()

    for candidate in _search_candidates(start):
        if _is_link_like(candidate) or not candidate.is_file():
            continue
        if guard.is_sensitive(candidate):
            continue
        try:
            _, text = _read_text_file(guard, candidate, guard.relative(candidate))
        except ToolExecutionError as exc:
            if exc.kind in {ErrorKind.FILE_TOO_LARGE, ErrorKind.FILE_NOT_TEXT}:
                continue
            raise

        for line_number, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.casefold()
            if needle not in haystack:
                continue
            matches.append(
                {
                    "path": guard.relative(candidate),
                    "line_number": line_number,
                    "preview": _bounded_text(line, SEARCH_PREVIEW_CHARS),
                }
            )
            if len(matches) >= max_results:
                truncated = True
                break
        if truncated:
            break

    return {
        "query": query,
        "path": guard.relative(start),
        "case_sensitive": case_sensitive,
        "matches": matches,
        "truncated": truncated,
    }


def edit_file(
    guard: WorkspaceGuard,
    *,
    path: str,
    old_text: str,
    new_text: str,
    expected_sha256: str,
) -> dict[str, Any]:
    target = guard.resolve_for_write(path)
    data, text = _read_text_file(guard, target, path, reject_symlink=True)
    before_sha = _sha256(data)
    expected = _normalize_sha256(expected_sha256)
    if before_sha != expected:
        raise ToolExecutionError(
            ErrorKind.STALE_FILE,
            f"file changed since it was read: {path}",
            retryable=True,
            metadata={"path": guard.relative(target), "current_sha256": before_sha},
        )
    if old_text == "":
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            "old_text must not be empty",
        )
    if old_text == new_text:
        raise ToolExecutionError(
            ErrorKind.NO_CHANGE,
            "old_text and new_text are identical",
        )

    first = text.find(old_text)
    if first < 0:
        raise ToolExecutionError(
            ErrorKind.EDIT_TEXT_NOT_FOUND,
            "old_text was not found exactly in the file",
            retryable=True,
        )
    if text.find(old_text, first + 1) >= 0:
        raise ToolExecutionError(
            ErrorKind.EDIT_TEXT_AMBIGUOUS,
            "old_text occurs more than once in the file",
            retryable=True,
        )

    candidate_text = text[:first] + new_text + text[first + len(old_text) :]
    candidate_data = _encode_text(guard, candidate_text)
    _atomic_replace(
        guard,
        target,
        candidate_data,
        expected_sha256=before_sha,
        original_mode=stat.S_IMODE(target.stat().st_mode),
    )
    after_sha = _sha256(candidate_data)
    diff = "".join(
        unified_diff(
            text.splitlines(keepends=True),
            candidate_text.splitlines(keepends=True),
            fromfile=f"a/{guard.relative(target)}",
            tofile=f"b/{guard.relative(target)}",
            n=2,
        )
    )
    return {
        "path": guard.relative(target),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "replacement_count": 1,
        "removed_lines": _line_span(old_text),
        "added_lines": _line_span(new_text),
        "diff_preview": _bounded_text(diff, DIFF_PREVIEW_CHARS),
    }


def write_file(
    guard: WorkspaceGuard,
    *,
    path: str,
    content: str,
    mode: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if mode not in {"create", "overwrite"}:
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            "mode must be 'create' or 'overwrite'",
        )
    target = guard.resolve_for_write(path)
    candidate_data = _encode_text(guard, content)

    if _is_link_like(target):
        raise ToolExecutionError(
            ErrorKind.PATH_IS_SYMLINK,
            f"writing through a symlink is denied: {path}",
        )

    if mode == "create":
        if expected_sha256 is not None:
            raise ToolExecutionError(
                ErrorKind.INVALID_ARGUMENTS,
                "expected_sha256 must be null for create",
            )
        if target.exists():
            raise ToolExecutionError(
                ErrorKind.FILE_ALREADY_EXISTS,
                f"create target already exists: {path}",
            )
        if not target.parent.exists():
            raise ToolExecutionError(
                ErrorKind.PATH_NOT_FOUND,
                f"parent directory does not exist: {guard.relative(target.parent)}",
            )
        if not target.parent.is_dir():
            raise ToolExecutionError(
                ErrorKind.PATH_WRITE_DENIED,
                f"parent path is not a directory: {guard.relative(target.parent)}",
            )
        _atomic_replace(guard, target, candidate_data, expected_sha256=None)
        before_sha: str | None = None
    else:
        if expected_sha256 is None:
            raise ToolExecutionError(
                ErrorKind.INVALID_ARGUMENTS,
                "expected_sha256 is required for overwrite",
            )
        current_data, _ = _read_text_file(
            guard,
            target,
            path,
            reject_symlink=True,
        )
        before_sha = _sha256(current_data)
        expected = _normalize_sha256(expected_sha256)
        if before_sha != expected:
            raise ToolExecutionError(
                ErrorKind.STALE_FILE,
                f"file changed since it was read: {path}",
                retryable=True,
                metadata={"path": guard.relative(target), "current_sha256": before_sha},
            )
        _atomic_replace(
            guard,
            target,
            candidate_data,
            expected_sha256=before_sha,
            original_mode=stat.S_IMODE(target.stat().st_mode),
        )

    return {
        "path": guard.relative(target),
        "mode": mode,
        "before_sha256": before_sha,
        "after_sha256": _sha256(candidate_data),
    }


def _walk_directory(
    root: Path,
    *,
    max_depth: int | None,
) -> Iterator[tuple[Path, int]]:
    def walk(directory: Path, depth: int) -> Iterator[tuple[Path, int]]:
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda item: (item.name.casefold(), item.name),
            )
        except OSError as exc:
            raise ToolExecutionError(
                ErrorKind.PATH_READ_DENIED,
                f"directory cannot be read: {directory.name}",
            ) from exc
        for child in children:
            if (
                _policy_component(child.name) in SKIPPED_DIRECTORIES
                and (_is_link_like(child) or child.is_dir())
            ):
                continue
            yield child, depth
            if (
                (max_depth is None or depth < max_depth)
                and not _is_link_like(child)
                and child.is_dir()
            ):
                yield from walk(child, depth + 1)

    yield from walk(root, 1)


def _search_candidates(start: Path) -> Iterator[Path]:
    if start.is_file() or _is_link_like(start):
        yield start
        return
    for candidate, _ in _walk_directory(start, max_depth=None):
        if _is_link_like(candidate) or candidate.is_file():
            yield candidate


def _list_entry(guard: WorkspaceGuard, path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ToolExecutionError(
            ErrorKind.PATH_READ_DENIED,
            f"file metadata cannot be read: {guard.relative(path)}",
        ) from exc
    return {"path": guard.relative(path), "type": "file", "size": size}


def _require_existing(path: Path, display_path: str) -> None:
    if not path.exists():
        raise ToolExecutionError(
            ErrorKind.PATH_NOT_FOUND,
            f"path does not exist: {display_path}",
        )


def _read_text_file(
    guard: WorkspaceGuard,
    target: Path,
    display_path: str,
    *,
    reject_symlink: bool = False,
) -> tuple[bytes, str]:
    if reject_symlink and _is_link_like(target):
        raise ToolExecutionError(
            ErrorKind.PATH_IS_SYMLINK,
            f"symlink files cannot be modified: {display_path}",
        )
    if not target.exists():
        raise ToolExecutionError(
            ErrorKind.PATH_NOT_FOUND,
            f"file does not exist: {display_path}",
        )
    if target.is_dir():
        raise ToolExecutionError(
            ErrorKind.PATH_IS_DIRECTORY,
            f"path is a directory, not a file: {display_path}",
        )
    if not target.is_file():
        raise ToolExecutionError(
            ErrorKind.PATH_READ_DENIED,
            f"path is not a regular file: {display_path}",
        )

    try:
        with target.open("rb") as stream:
            data = stream.read(guard.max_file_bytes + 1)
    except OSError as exc:
        raise ToolExecutionError(
            ErrorKind.PATH_READ_DENIED,
            f"file cannot be read: {display_path}",
        ) from exc
    if len(data) > guard.max_file_bytes:
        raise ToolExecutionError(
            ErrorKind.FILE_TOO_LARGE,
            f"file exceeds the {guard.max_file_bytes}-byte limit: {display_path}",
        )
    if b"\x00" in data:
        raise ToolExecutionError(
            ErrorKind.FILE_NOT_TEXT,
            f"file contains NUL bytes: {display_path}",
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError(
            ErrorKind.FILE_NOT_TEXT,
            f"file is not valid UTF-8 text: {display_path}",
        ) from exc
    return data, text


def _encode_text(guard: WorkspaceGuard, content: str) -> bytes:
    if "\x00" in content:
        raise ToolExecutionError(
            ErrorKind.FILE_NOT_TEXT,
            "text content must not contain NUL characters",
        )
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolExecutionError(
            ErrorKind.FILE_NOT_TEXT,
            "content cannot be encoded as UTF-8",
        ) from exc
    if len(data) > guard.max_file_bytes:
        raise ToolExecutionError(
            ErrorKind.FILE_TOO_LARGE,
            f"content exceeds the {guard.max_file_bytes}-byte limit",
        )
    return data


def _atomic_replace(
    guard: WorkspaceGuard,
    target: Path,
    data: bytes,
    *,
    expected_sha256: str | None,
    original_mode: int | None = None,
) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        if original_mode is not None:
            os.chmod(temp_path, original_mode)

        if expected_sha256 is None:
            if target.exists() or target.is_symlink():
                raise ToolExecutionError(
                    ErrorKind.FILE_ALREADY_EXISTS,
                    f"create target already exists: {guard.relative(target)}",
                )
            try:
                if os.name == "nt":
                    os.rename(temp_path, target)
                    temp_path = None
                else:
                    os.link(temp_path, target)
            except FileExistsError as exc:
                display_path = target.relative_to(guard.root).as_posix()
                raise ToolExecutionError(
                    ErrorKind.FILE_ALREADY_EXISTS,
                    f"create target already exists: {display_path}",
                ) from exc
            return
        else:
            current_data, _ = _read_text_file(
                guard,
                target,
                guard.relative(target),
                reject_symlink=True,
            )
            if _sha256(current_data) != expected_sha256:
                raise ToolExecutionError(
                    ErrorKind.STALE_FILE,
                    f"file changed before atomic replacement: {guard.relative(target)}",
                    retryable=True,
                    metadata={
                        "path": guard.relative(target),
                        "current_sha256": _sha256(current_data),
                    },
                )
            os.replace(temp_path, target)
            temp_path = None
    except ToolExecutionError:
        raise
    except OSError as exc:
        raise ToolExecutionError(
            ErrorKind.PATH_WRITE_DENIED,
            f"atomic write failed for {guard.relative(target)}: {type(exc).__name__}",
        ) from exc
    finally:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)


def _normalize_sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            "expected_sha256 must be a 64-character hexadecimal digest",
        )
    normalized = value.casefold()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            "expected_sha256 must be a 64-character hexadecimal digest",
        )
    return normalized


def _require_range(name: str, value: int, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise ToolExecutionError(
            ErrorKind.INVALID_ARGUMENTS,
            f"{name} must be between {minimum} and {maximum}",
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line_span(text: str) -> int:
    return len(text.splitlines()) if text else 0


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    marker = ""
    for _ in range(2):
        omitted = len(text) - head - tail
        marker = f"\n... <{omitted} characters omitted> ...\n"
        available = max(limit - len(marker), 2)
        head = available // 2
        tail = available - head
    omitted = len(text) - head - tail
    marker = f"\n... <{omitted} characters omitted> ...\n"
    return text[:head] + marker + text[-tail:]


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _policy_component(value: str) -> str:
    """Normalize names the way common Windows paths ignore trailing dots/spaces."""

    return value.rstrip(" .").casefold()


def _has_protected_metadata_component(*component_groups: tuple[str, ...]) -> bool:
    return any(
        _policy_component(part) in _PROTECTED_METADATA_COMPONENTS
        for components in component_groups
        for part in components
    )
