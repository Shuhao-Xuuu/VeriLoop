from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

import veriloop.filesystem as filesystem
from veriloop.filesystem import DEFAULT_MAX_FILE_BYTES, WorkspaceGuard
from veriloop.protocol import ErrorKind, ToolCall
from veriloop.tools import ToolExecutionError, ToolRegistry, register_filesystem_tools


def make_registry(workspace: Path) -> tuple[WorkspaceGuard, ToolRegistry]:
    guard = WorkspaceGuard(workspace)
    registry = ToolRegistry()
    register_filesystem_tools(registry, guard)
    return guard, registry


def execute(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, object],
    *,
    call_id: str = "call-1",
):
    return registry.execute(ToolCall(id=call_id, name=name, arguments=arguments))


def content(result) -> dict[str, object]:
    return json.loads(result.content)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def create_symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable on this host: {exc}")


def test_workspace_guard_accepts_nested_and_missing_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "src"
    nested.mkdir(parents=True)
    guard = WorkspaceGuard(workspace)

    assert guard.resolve("src/module.py") == nested / "module.py"
    assert guard.resolve("./src//module.py") == nested / "module.py"
    assert guard.root == workspace.resolve()


@pytest.mark.parametrize(
    "model_path",
    [
        "../outside.txt",
        "../../outside.txt",
        "C:\\outside.txt",
        "C:outside.txt",
    ],
)
def test_workspace_guard_rejects_escape_and_windows_paths(
    tmp_path: Path, model_path: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    guard = WorkspaceGuard(workspace)

    with pytest.raises(ToolExecutionError) as captured:
        guard.resolve(model_path)

    assert captured.value.kind is ErrorKind.PATH_OUTSIDE_WORKSPACE


@pytest.mark.parametrize("model_path", [".env::$DATA", "notes.txt:stream"])
def test_workspace_guard_rejects_windows_alternate_data_stream_paths(
    tmp_path: Path, model_path: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    guard = WorkspaceGuard(workspace)

    with pytest.raises(ToolExecutionError) as captured:
        guard.resolve(model_path)

    assert captured.value.kind is ErrorKind.INVALID_ARGUMENTS


def test_workspace_guard_rejects_absolute_and_prefix_deception(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    sibling = tmp_path / "workspace_evil"
    workspace.mkdir()
    sibling.mkdir()
    guard = WorkspaceGuard(workspace)

    for model_path in (str(sibling / "file.txt"), "../workspace_evil/file.txt"):
        with pytest.raises(ToolExecutionError) as captured:
            guard.resolve(model_path)
        assert captured.value.kind is ErrorKind.PATH_OUTSIDE_WORKSPACE


def test_workspace_root_must_exist_and_be_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        WorkspaceGuard(tmp_path / "missing")
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        WorkspaceGuard(file_path)


def test_read_missing_directory_and_sensitive_path_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("SECRET=fake", encoding="utf-8")
    (workspace / "folder").mkdir()
    _, registry = make_registry(workspace)

    missing = execute(registry, "read_file", {"path": "missing.txt"})
    directory = execute(registry, "read_file", {"path": "folder"})
    protected = execute(registry, "read_file", {"path": ".env"})

    assert missing.error_kind is ErrorKind.PATH_NOT_FOUND
    assert directory.error_kind is ErrorKind.PATH_IS_DIRECTORY
    assert protected.error_kind is ErrorKind.PATH_READ_DENIED


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (".git/config", "git metadata"),
        (".veriloop/state.json", "VeriLoop metadata"),
        (".GIT/config", "case alias"),
    ],
)
def test_read_file_rejects_protected_metadata_directories(
    tmp_path: Path,
    path: str,
    payload: str,
) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / path
    target.parent.mkdir(parents=True)
    target.write_text(payload, encoding="utf-8")
    _, registry = make_registry(workspace)

    result = execute(registry, "read_file", {"path": path})

    assert result.error_kind is ErrorKind.PATH_READ_DENIED
    assert payload not in result.content


@pytest.mark.parametrize(
    "path",
    [".git./config", ".git /config", ".VERILOOP./state.json"],
)
def test_read_file_rejects_lexical_protected_directory_aliases(
    tmp_path: Path,
    path: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry = make_registry(workspace)

    result = execute(registry, "read_file", {"path": path})

    assert result.error_kind is ErrorKind.PATH_READ_DENIED


def test_read_file_rejects_canonical_alias_into_protected_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    protected = workspace / ".git"
    protected.mkdir(parents=True)
    (protected / "config").write_text("hidden metadata", encoding="utf-8")
    create_symlink_or_skip(workspace / "metadata", protected, directory=True)
    _, registry = make_registry(workspace)

    result = execute(registry, "read_file", {"path": "metadata/config"})

    assert result.error_kind is ErrorKind.PATH_READ_DENIED
    assert "hidden metadata" not in result.content


def test_protected_component_matching_does_not_reject_ordinary_names(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".gitignore").write_text("ordinary", encoding="utf-8")
    (workspace / "veriloop_notes.txt").write_text("ordinary", encoding="utf-8")
    _, registry = make_registry(workspace)

    assert execute(registry, "read_file", {"path": ".gitignore"}).ok
    assert execute(registry, "read_file", {"path": "veriloop_notes.txt"}).ok


def test_sensitive_matching_is_component_aware(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "src"
    source.mkdir(parents=True)
    (source / "target.py").write_text("key = 'not a secret filename'", encoding="utf-8")
    (source / "my.env.example.txt").write_text("ordinary", encoding="utf-8")
    (source / ".env.local").write_text("protected", encoding="utf-8")
    _, registry = make_registry(workspace)

    assert execute(registry, "read_file", {"path": "src/target.py"}).ok
    assert execute(registry, "read_file", {"path": "src/my.env.example.txt"}).ok
    assert (
        execute(registry, "read_file", {"path": "src/.env.local"}).error_kind
        is ErrorKind.PATH_READ_DENIED
    )


def test_read_file_range_utf8_sha_and_truncation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw = "alpha\n第二行\ngamma\ndelta\n".encode()
    (workspace / "notes.txt").write_bytes(raw)
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "read_file",
        {"path": "notes.txt", "start_line": 2, "end_line": 3},
    )
    payload = content(result)

    assert result.ok
    assert payload["path"] == "notes.txt"
    assert payload["requested_range"] == {"start_line": 2, "end_line": 3}
    assert payload["actual_range"] == {"start_line": 2, "end_line": 3}
    assert payload["total_lines"] == 4
    assert payload["sha256"] == sha256(raw)
    assert "第二行" in payload["content"]
    assert payload["truncated"] is True
    assert content(execute(registry, "read_file", {"path": "notes.txt"}))["sha256"] == sha256(raw)


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "lines.txt", "start_line": 0, "end_line": 1},
        {"path": "lines.txt", "start_line": 3, "end_line": 2},
        {"path": "lines.txt", "start_line": 1, "end_line": 501},
    ],
)
def test_read_file_rejects_invalid_line_ranges(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "lines.txt").write_text("a\nb\nc", encoding="utf-8")
    _, registry = make_registry(workspace)

    result = execute(registry, "read_file", arguments)

    assert result.error_kind is ErrorKind.INVALID_ARGUMENTS


@pytest.mark.parametrize("path", ["nested/../inside.txt", "missing/../inside.txt"])
def test_workspace_guard_rejects_parent_components_even_if_result_is_inside(
    tmp_path: Path,
    path: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inside.txt").write_text("inside", encoding="utf-8")
    guard = WorkspaceGuard(workspace)

    with pytest.raises(ToolExecutionError) as captured:
        guard.resolve(path)

    assert captured.value.kind is ErrorKind.PATH_OUTSIDE_WORKSPACE


@pytest.mark.parametrize(
    ("name", "data", "kind"),
    [
        ("nul.bin", b"hello\x00world", ErrorKind.FILE_NOT_TEXT),
        ("invalid.bin", b"\xff\xfe", ErrorKind.FILE_NOT_TEXT),
        ("large.txt", b"x" * (DEFAULT_MAX_FILE_BYTES + 1), ErrorKind.FILE_TOO_LARGE),
    ],
    ids=["nul", "invalid-utf8", "oversized"],
)
def test_read_file_rejects_non_text_and_oversized_files(
    tmp_path: Path, name: str, data: bytes, kind: ErrorKind
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / name).write_bytes(data)
    _, registry = make_registry(workspace)

    result = execute(registry, "read_file", {"path": name})

    assert result.error_kind is kind


def test_list_files_is_sorted_depth_bounded_and_skips_caches(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "folder" / "deep").mkdir(parents=True)
    (workspace / "folder" / "item.txt").write_text("item", encoding="utf-8")
    (workspace / "folder" / "deep" / "hidden.txt").write_text("deep", encoding="utf-8")
    (workspace / "b.txt").write_text("b", encoding="utf-8")
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "cache.pyc").write_bytes(b"cache")
    _, registry = make_registry(workspace)

    first = content(
        execute(
            registry,
            "list_files",
            {"path": ".", "max_depth": 1, "max_results": 100},
        )
    )
    second = content(
        execute(
            registry,
            "list_files",
            {"path": ".", "max_depth": 1, "max_results": 100},
        )
    )
    paths = [entry["path"] for entry in first["entries"]]

    assert first == second
    assert paths == ["a.txt", "b.txt", "folder"]
    assert "folder/item.txt" not in paths
    assert all("__pycache__" not in item for item in paths)


def test_list_files_max_results_sets_truncated(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(4):
        (workspace / f"{index}.txt").write_text(str(index), encoding="utf-8")
    _, registry = make_registry(workspace)

    payload = content(
        execute(
            registry,
            "list_files",
            {"max_depth": 2, "max_results": 2},
        )
    )

    assert [entry["path"] for entry in payload["entries"]] == ["0.txt", "1.txt"]
    assert payload["truncated"] is True


def test_list_and_search_reject_explicit_git_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    git_dir = workspace / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "config").write_text("hidden needle", encoding="utf-8")
    _, registry = make_registry(workspace)

    listed = execute(registry, "list_files", {"path": ".git"})
    searched = execute(
        registry,
        "search_text",
        {"path": ".git", "query": "needle"},
    )

    assert listed.error_kind is ErrorKind.PATH_READ_DENIED
    assert searched.error_kind is ErrorKind.PATH_READ_DENIED


def test_skip_names_do_not_hide_regular_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "build").write_text("visible needle", encoding="utf-8")
    _, registry = make_registry(workspace)

    listed = content(execute(registry, "list_files", {}))
    searched = content(execute(registry, "search_text", {"query": "needle"}))

    assert [item["path"] for item in listed["entries"]] == ["build"]
    assert [item["path"] for item in searched["matches"]] == ["build"]


def test_list_and_search_skip_named_directory_symlinks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "ordinary"
    target.mkdir()
    (target / "item.txt").write_text("hidden needle", encoding="utf-8")
    create_symlink_or_skip(workspace / "build", target, directory=True)
    _, registry = make_registry(workspace)

    listed = content(execute(registry, "list_files", {}))
    searched = content(execute(registry, "search_text", {"query": "needle"}))

    assert [item["path"] for item in listed["entries"]] == [
        "ordinary",
        "ordinary/item.txt",
    ]
    assert [item["path"] for item in searched["matches"]] == [
        "ordinary/item.txt"
    ]


def test_search_text_literal_case_options_and_order(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "b.txt").write_text("Needle two\n", encoding="utf-8")
    (workspace / "a.txt").write_text("needle one\nother\n", encoding="utf-8")
    _, registry = make_registry(workspace)

    insensitive = content(execute(registry, "search_text", {"query": "needle"}))
    sensitive = content(
        execute(
            registry,
            "search_text",
            {"query": "needle", "case_sensitive": True},
        )
    )

    assert [(item["path"], item["line_number"]) for item in insensitive["matches"]] == [
        ("a.txt", 1),
        ("b.txt", 1),
    ]
    assert [item["path"] for item in sensitive["matches"]] == ["a.txt"]


def test_search_text_empty_query_limit_and_skipped_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("hit\nhit\n", encoding="utf-8")
    (workspace / "binary.bin").write_bytes(b"hit\x00")
    (workspace / "large.txt").write_bytes(b"hit" + b"x" * DEFAULT_MAX_FILE_BYTES)
    _, registry = make_registry(workspace)

    empty = execute(registry, "search_text", {"query": ""})
    limited = content(
        execute(registry, "search_text", {"query": "hit", "max_results": 1})
    )

    assert empty.error_kind is ErrorKind.INVALID_ARGUMENTS
    assert len(limited["matches"]) == 1
    assert limited["matches"][0]["path"] == "a.txt"
    assert limited["truncated"] is True


def test_search_preview_is_strictly_bounded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "long.txt").write_text(
        "HEAD-needle-" + ("x" * 1000) + "-TAIL",
        encoding="utf-8",
    )
    _, registry = make_registry(workspace)

    searched = content(execute(registry, "search_text", {"query": "needle"}))
    preview = searched["matches"][0]["preview"]

    assert len(preview) <= filesystem.SEARCH_PREVIEW_CHARS
    assert preview.startswith("HEAD-needle")
    assert preview.endswith("-TAIL")
    assert "characters omitted" in preview


def test_edit_file_unique_replacement_changes_sha_and_preserves_mode(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "module.py"
    before = b"def value():\n    return 1\n"
    target.write_bytes(before)
    if os.name != "nt":
        target.chmod(0o640)
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "edit_file",
        {
            "path": "module.py",
            "old_text": "return 1",
            "new_text": "return 2",
            "expected_sha256": sha256(before),
        },
    )
    payload = content(result)

    assert result.ok
    assert target.read_text(encoding="utf-8") == "def value():\n    return 2\n"
    assert payload["before_sha256"] == sha256(before)
    assert payload["after_sha256"] == sha256(target.read_bytes())
    assert payload["before_sha256"] != payload["after_sha256"]
    assert payload["replacement_count"] == 1
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.parametrize(
    ("old_text", "new_text", "expected", "kind"),
    [
        ("missing", "new", "CURRENT", ErrorKind.EDIT_TEXT_NOT_FOUND),
        ("same", "changed", "CURRENT", ErrorKind.EDIT_TEXT_AMBIGUOUS),
        ("", "new", "CURRENT", ErrorKind.INVALID_ARGUMENTS),
        ("unique", "unique", "CURRENT", ErrorKind.NO_CHANGE),
        ("unique", "changed", "0" * 64, ErrorKind.STALE_FILE),
    ],
)
def test_edit_file_failures_leave_original_bytes_unchanged(
    tmp_path: Path,
    old_text: str,
    new_text: str,
    expected: str,
    kind: ErrorKind,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "data.txt"
    original = b"same same unique\n"
    target.write_bytes(original)
    _, registry = make_registry(workspace)
    digest = sha256(original) if expected == "CURRENT" else expected

    result = execute(
        registry,
        "edit_file",
        {
            "path": "data.txt",
            "old_text": old_text,
            "new_text": new_text,
            "expected_sha256": digest,
        },
    )

    assert result.error_kind is kind
    assert target.read_bytes() == original


def test_edit_file_detects_overlapping_ambiguous_matches(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "overlap.txt"
    target.write_text("aaa", encoding="utf-8")
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "edit_file",
        {
            "path": "overlap.txt",
            "old_text": "aa",
            "new_text": "b",
            "expected_sha256": sha256(b"aaa"),
        },
    )

    assert result.error_kind is ErrorKind.EDIT_TEXT_AMBIGUOUS
    assert target.read_bytes() == b"aaa"


def test_atomic_replace_uses_same_directory(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "data.txt"
    target.write_text("before", encoding="utf-8")
    original_replace = filesystem.os.replace
    calls: list[tuple[Path, Path]] = []

    def recording_replace(source, destination) -> None:
        calls.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(filesystem.os, "replace", recording_replace)
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "edit_file",
        {
            "path": "data.txt",
            "old_text": "before",
            "new_text": "after",
            "expected_sha256": sha256(b"before"),
        },
    )

    assert result.ok
    assert len(calls) == 1
    assert calls[0][0].parent == target.parent
    assert calls[0][1] == target


def test_atomic_replace_failure_cleans_temp_and_preserves_file(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "data.txt"
    target.write_text("before", encoding="utf-8")
    before_names = {item.name for item in workspace.iterdir()}

    def fail_replace(source, destination) -> None:
        raise OSError("fictional replace failure")

    monkeypatch.setattr(filesystem.os, "replace", fail_replace)
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "edit_file",
        {
            "path": "data.txt",
            "old_text": "before",
            "new_text": "after",
            "expected_sha256": sha256(b"before"),
        },
    )

    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert target.read_bytes() == b"before"
    assert {item.name for item in workspace.iterdir()} == before_names


def test_edit_rechecks_sha_immediately_before_replace(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "data.txt"
    original = b"before"
    external = b"external change"
    target.write_bytes(original)
    original_chmod = filesystem.os.chmod

    def change_target_after_temp_write(path, mode) -> None:
        original_chmod(path, mode)
        target.write_bytes(external)

    monkeypatch.setattr(filesystem.os, "chmod", change_target_after_temp_write)
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "edit_file",
        {
            "path": "data.txt",
            "old_text": "before",
            "new_text": "after",
            "expected_sha256": sha256(original),
        },
    )

    assert result.error_kind is ErrorKind.STALE_FILE
    assert target.read_bytes() == external
    assert [item.name for item in workspace.iterdir()] == ["data.txt"]


def test_write_file_create_and_overwrite(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry = make_registry(workspace)

    created = execute(
        registry,
        "write_file",
        {"path": "new.txt", "content": "first", "mode": "create"},
    )
    created_payload = content(created)
    overwritten = execute(
        registry,
        "write_file",
        {
            "path": "new.txt",
            "content": "second",
            "mode": "overwrite",
            "expected_sha256": created_payload["after_sha256"],
        },
    )

    assert created.ok and overwritten.ok
    assert created_payload["before_sha256"] is None
    assert content(overwritten)["before_sha256"] == created_payload["after_sha256"]
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "second"


def test_write_file_create_and_overwrite_failures(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "existing.txt"
    target.write_text("original", encoding="utf-8")
    _, registry = make_registry(workspace)

    exists = execute(
        registry,
        "write_file",
        {"path": "existing.txt", "content": "new", "mode": "create"},
    )
    no_sha = execute(
        registry,
        "write_file",
        {"path": "existing.txt", "content": "new", "mode": "overwrite"},
    )
    stale = execute(
        registry,
        "write_file",
        {
            "path": "existing.txt",
            "content": "new",
            "mode": "overwrite",
            "expected_sha256": "0" * 64,
        },
    )
    missing_parent = execute(
        registry,
        "write_file",
        {"path": "missing/new.txt", "content": "new", "mode": "create"},
    )

    assert exists.error_kind is ErrorKind.FILE_ALREADY_EXISTS
    assert no_sha.error_kind is ErrorKind.INVALID_ARGUMENTS
    assert stale.error_kind is ErrorKind.STALE_FILE
    assert missing_parent.error_kind is ErrorKind.PATH_NOT_FOUND
    assert target.read_text(encoding="utf-8") == "original"


def test_write_create_does_not_clobber_target_appearing_at_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "new.txt"
    external = b"external winner"
    original_replace = filesystem.os.replace

    def inject_competing_target(destination) -> None:
        destination_path = Path(destination)
        if not destination_path.exists():
            destination_path.write_bytes(external)

    def racing_replace(source, destination) -> None:
        inject_competing_target(destination)
        original_replace(source, destination)

    monkeypatch.setattr(filesystem.os, "replace", racing_replace)
    if os.name == "nt":
        original_install = filesystem.os.rename

        def racing_install(source, destination) -> None:
            inject_competing_target(destination)
            original_install(source, destination)

        monkeypatch.setattr(filesystem.os, "rename", racing_install)
    else:
        original_install = filesystem.os.link

        def racing_install(source, destination) -> None:
            inject_competing_target(destination)
            original_install(source, destination)

        monkeypatch.setattr(filesystem.os, "link", racing_install)

    _, registry = make_registry(workspace)
    result = execute(
        registry,
        "write_file",
        {"path": "new.txt", "content": "candidate", "mode": "create"},
    )

    assert result.error_kind is ErrorKind.FILE_ALREADY_EXISTS
    assert target.read_bytes() == external
    assert [item.name for item in workspace.iterdir()] == ["new.txt"]


@pytest.mark.parametrize("path", [".git/config", ".veriloop/state.json", ".env", "key.pem"])
def test_write_file_rejects_protected_paths(tmp_path: Path, path: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "write_file",
        {"path": path, "content": "value", "mode": "create"},
    )

    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (workspace / path).exists()


def test_write_file_rejects_nul_and_oversized_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, registry = make_registry(workspace)

    nul = execute(
        registry,
        "write_file",
        {"path": "nul.txt", "content": "a\x00b", "mode": "create"},
    )
    large = execute(
        registry,
        "write_file",
        {
            "path": "large.txt",
            "content": "x" * (DEFAULT_MAX_FILE_BYTES + 1),
            "mode": "create",
        },
    )

    assert nul.error_kind is ErrorKind.FILE_NOT_TEXT
    assert large.error_kind is ErrorKind.FILE_TOO_LARGE
    assert list(workspace.iterdir()) == []


def test_external_symlink_is_rejected_for_read_edit_and_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = workspace / "link.txt"
    create_symlink_or_skip(link, outside)
    _, registry = make_registry(workspace)

    read = execute(registry, "read_file", {"path": "link.txt"})
    edit = execute(
        registry,
        "edit_file",
        {
            "path": "link.txt",
            "old_text": "outside",
            "new_text": "changed",
            "expected_sha256": sha256(b"outside"),
        },
    )
    overwrite = execute(
        registry,
        "write_file",
        {
            "path": "link.txt",
            "content": "changed",
            "mode": "overwrite",
            "expected_sha256": sha256(b"outside"),
        },
    )

    assert read.error_kind is ErrorKind.PATH_OUTSIDE_WORKSPACE
    assert edit.error_kind is ErrorKind.PATH_OUTSIDE_WORKSPACE
    assert overwrite.error_kind is ErrorKind.PATH_OUTSIDE_WORKSPACE
    assert outside.read_text(encoding="utf-8") == "outside"


def test_list_and_search_do_not_follow_external_symlink_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("unique-outside-needle", encoding="utf-8")
    link = workspace / "linked"
    create_symlink_or_skip(link, outside, directory=True)
    _, registry = make_registry(workspace)

    listed = content(execute(registry, "list_files", {}))
    searched = content(
        execute(registry, "search_text", {"query": "unique-outside-needle"})
    )

    assert listed["entries"] == [{"path": "linked", "type": "symlink"}]
    assert searched["matches"] == []


def test_symlink_directory_cannot_be_used_to_create_outside(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "linked"
    create_symlink_or_skip(link, outside, directory=True)
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "write_file",
        {"path": "linked/new.txt", "content": "no", "mode": "create"},
    )

    assert result.error_kind is ErrorKind.PATH_OUTSIDE_WORKSPACE
    assert not (outside / "new.txt").exists()


def test_protected_lexical_directory_symlink_cannot_bypass_write_rule(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ordinary = workspace / "ordinary"
    ordinary.mkdir()
    create_symlink_or_skip(workspace / ".git", ordinary, directory=True)
    _, registry = make_registry(workspace)

    result = execute(
        registry,
        "write_file",
        {"path": ".git/config", "content": "blocked", "mode": "create"},
    )

    assert result.error_kind is ErrorKind.PATH_WRITE_DENIED
    assert not (ordinary / "config").exists()


def test_list_and_search_reject_intermediate_symlink_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    real = workspace / "real" / "sub"
    real.mkdir(parents=True)
    (real / "item.txt").write_text("needle", encoding="utf-8")
    create_symlink_or_skip(workspace / "linked", workspace / "real", directory=True)
    _, registry = make_registry(workspace)

    listed = execute(registry, "list_files", {"path": "linked/sub"})
    searched = execute(
        registry,
        "search_text",
        {"path": "linked/sub", "query": "needle"},
    )

    assert listed.error_kind is ErrorKind.PATH_READ_DENIED
    assert searched.error_kind is ErrorKind.PATH_READ_DENIED
