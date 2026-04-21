"""Filesystem-backed storage for user custom model templates."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CUSTOM_MODEL_SCHEMA_VERSION = "0.1"
DEFAULT_RETENTION_POLICY = {
    "mode": "count",
    "max_count": 40,
    "max_age_days": 90,
    "min_count_per_model": 3,
}
RETENTION_POLICY_FILENAME = ".retention-policy.json"


class CustomModelStore:
    """Persist custom model entries as JSON files in the local user data dir."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.trash_dir = self.root_dir / ".trash"
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        self.git_executable = shutil.which("git")
        self._ensure_repo()

    @property
    def policy_path(self) -> Path:
        return self.root_dir / RETENTION_POLICY_FILENAME

    @property
    def history_backend(self) -> str:
        return "git" if self.git_executable else "none"

    def _run_git(
        self,
        args: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str] | None:
        if self.git_executable is None:
            return None
        return subprocess.run(
            [self.git_executable, *args],
            cwd=str(self.root_dir),
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def _ensure_repo(self) -> None:
        if self.git_executable is None:
            return
        git_dir = self.root_dir / ".git"
        if not git_dir.exists():
            init_result = self._run_git(["init", "-b", "main"], check=False)
            if init_result is None or init_result.returncode != 0:
                self._run_git(["init"])
        self._run_git(["config", "user.name", "MedFusion Custom Model Store"])
        self._run_git(
            ["config", "user.email", "custom-model-store@medfusion.local"]
        )
        exclude_path = self.root_dir / ".git" / "info" / "exclude"
        existing_exclude = ""
        if exclude_path.exists():
            existing_exclude = exclude_path.read_text(encoding="utf-8")
        if RETENTION_POLICY_FILENAME not in existing_exclude:
            exclude_path.write_text(
                f"{existing_exclude}\n{RETENTION_POLICY_FILENAME}\n".lstrip(),
                encoding="utf-8",
            )
        if not self.policy_path.exists():
            self.policy_path.write_text(
                json.dumps(DEFAULT_RETENTION_POLICY, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _entry_path(self, entry_id: str) -> Path:
        safe_id = "".join(
            character
            for character in entry_id
            if character.isalnum() or character in {"-", "_"}
        )
        return self.root_dir / f"{safe_id or 'custom-model'}.json"

    def _trash_entry_path(self, entry_id: str) -> Path:
        safe_id = "".join(
            character
            for character in entry_id
            if character.isalnum() or character in {"-", "_"}
        )
        return self.trash_dir / f"{safe_id or 'custom-model'}.json"

    def _relative_entry_path(self, entry_id: str) -> str:
        return self._entry_path(entry_id).name

    def get_retention_policy(self) -> dict[str, Any]:
        if not self.policy_path.exists():
            return dict(DEFAULT_RETENTION_POLICY)
        try:
            payload = json.loads(self.policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return dict(DEFAULT_RETENTION_POLICY)
        if not isinstance(payload, dict):
            return dict(DEFAULT_RETENTION_POLICY)
        return {
            "mode": payload.get("mode", DEFAULT_RETENTION_POLICY["mode"]),
            "max_count": int(payload.get("max_count", DEFAULT_RETENTION_POLICY["max_count"])),
            "max_age_days": int(
                payload.get("max_age_days", DEFAULT_RETENTION_POLICY["max_age_days"])
            ),
            "min_count_per_model": int(
                payload.get(
                    "min_count_per_model",
                    DEFAULT_RETENTION_POLICY["min_count_per_model"],
                )
            ),
        }

    def save_retention_policy(self, policy: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "mode": policy.get("mode", DEFAULT_RETENTION_POLICY["mode"]),
            "max_count": max(
                1,
                min(int(policy.get("max_count", DEFAULT_RETENTION_POLICY["max_count"])), 40),
            ),
            "max_age_days": max(
                1,
                int(policy.get("max_age_days", DEFAULT_RETENTION_POLICY["max_age_days"])),
            ),
            "min_count_per_model": max(
                1,
                int(
                    policy.get(
                        "min_count_per_model",
                        DEFAULT_RETENTION_POLICY["min_count_per_model"],
                    )
                ),
            ),
        }
        self.policy_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return normalized

    def _list_all_commits(self) -> list[str]:
        if self.git_executable is None:
            return []
        result = self._run_git(["rev-list", "--reverse", "HEAD"], check=False)
        if result is None or result.returncode != 0 or not result.stdout.strip():
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _entry_history_commits(self, entry_filename: str) -> list[str]:
        if self.git_executable is None:
            return []
        result = self._run_git(["log", "--format=%H", "--", entry_filename], check=False)
        if result is None or result.returncode != 0 or not result.stdout.strip():
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _retained_commits(self, policy: dict[str, Any]) -> list[str]:
        commits = self._list_all_commits()
        if not commits:
            return []
        mode = policy.get("mode", "count")
        if mode == "time":
            result = self._run_git(
                [
                    "rev-list",
                    "--reverse",
                    f"--since={int(policy.get('max_age_days', 90))} days ago",
                    "HEAD",
                ],
                check=False,
            )
            if result is not None and result.returncode == 0 and result.stdout.strip():
                retained = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if retained:
                    return retained
            return [commits[-1]]
        max_count = max(1, min(int(policy.get("max_count", 40)), 40))
        min_count_per_model = max(1, int(policy.get("min_count_per_model", 3)))

        mandatory_newest: list[str] = []
        seen_mandatory: set[str] = set()
        # Only active models participate in the per-model retention floor.
        # Deleted models in the recycle bin keep history, but do not reserve
        # guaranteed slots when the global cap is enforced.
        for path in sorted(self.root_dir.glob("*.json")):
            if path.name.startswith("."):
                continue
            entry_commits = self._entry_history_commits(path.name)[:min_count_per_model]
            for commit in entry_commits:
                if commit not in seen_mandatory:
                    seen_mandatory.add(commit)
                    mandatory_newest.append(commit)

        selected_newest = list(mandatory_newest)
        selected_set = set(selected_newest)
        effective_max_count = max(max_count, len(selected_set))
        for commit in reversed(commits):
            if commit in selected_set:
                continue
            if len(selected_newest) >= effective_max_count:
                break
            selected_newest.append(commit)
            selected_set.add(commit)

        return [commit for commit in commits if commit in selected_set]

    def _rewrite_history(self, retained_commits: list[str]) -> None:
        if self.git_executable is None or not retained_commits:
            return
        branch_result = self._run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
        )
        current_branch = (
            branch_result.stdout.strip()
            if branch_result and branch_result.returncode == 0 and branch_result.stdout.strip()
            else "main"
        )
        retained_count = len(retained_commits)
        original_policy = self.get_retention_policy()
        with tempfile.TemporaryDirectory(prefix="medfusion-custom-model-backup-") as backup_dir:
            backup_path = Path(backup_dir) / "repo-backup"
            self._run_git(["clone", str(self.root_dir), str(backup_path)], check=False)

            self._run_git(["checkout", "--orphan", "__retention_rewrite__"])
            for child in self.root_dir.iterdir():
                if child.name == ".git":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

            earliest = retained_commits[0]
            self._run_git(["checkout", earliest, "--", "."])
            self._run_git(["add", "-A", "."])
            self._run_git(
                [
                    "commit",
                    "-m",
                    f"custom-model history retained base {earliest[:12]}",
                ]
            )
            for commit in retained_commits[1:]:
                self._run_git(["cherry-pick", "--allow-empty", commit])
            self._run_git(["branch", "-M", current_branch])
            self.save_retention_policy(original_policy)
            self._run_git(["gc", "--prune=now"], check=False)

    def _commit_if_needed(self, message: str) -> None:
        if self.git_executable is None:
            return
        self._run_git(["add", "-A", "."])
        diff_result = self._run_git(["diff", "--cached", "--quiet"], check=False)
        if diff_result is not None and diff_result.returncode == 0:
            return
        self._run_git(["commit", "-m", message])
        # Keep the local history repo lightweight on long-lived client machines.
        self._run_git(["gc", "--auto"], check=False)
        self._apply_retention_policy()

    def _apply_retention_policy(self) -> None:
        if self.git_executable is None:
            return
        commits = self._list_all_commits()
        if len(commits) <= 1:
            return
        policy = self.get_retention_policy()
        retained = self._retained_commits(policy)
        if len(retained) >= len(commits):
            return
        self._rewrite_history(retained)

    def list_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(self.root_dir.glob("*.json")):
            if path.name.startswith("."):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload.setdefault("schema_version", CUSTOM_MODEL_SCHEMA_VERSION)
                    entries.append(payload)
            except json.JSONDecodeError:
                continue
        return entries

    def list_deleted_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(self.trash_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload.setdefault("schema_version", CUSTOM_MODEL_SCHEMA_VERSION)
                    payload.setdefault("deleted", True)
                    entries.append(payload)
            except json.JSONDecodeError:
                continue
        return entries

    def save_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry_id = str(entry["id"])
        path = self._entry_path(entry_id)
        persisted_entry = {
            **entry,
            "schema_version": entry.get("schema_version", CUSTOM_MODEL_SCHEMA_VERSION),
        }
        path.write_text(
            json.dumps(persisted_entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._commit_if_needed(f"custom-model save {entry_id}")
        return persisted_entry

    def delete_entry(self, entry_id: str) -> None:
        path = self._entry_path(entry_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["deleted"] = True
            payload["deleted_at"] = datetime.now(UTC).isoformat()
            trash_path = self._trash_entry_path(entry_id)
            trash_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            path.unlink()
            self._commit_if_needed(f"custom-model delete {entry_id}")

    def restore_deleted_entry(self, entry_id: str) -> dict[str, Any]:
        trash_path = self._trash_entry_path(entry_id)
        if not trash_path.exists():
            raise FileNotFoundError(f"回收站中不存在该模型: {entry_id}")
        payload = json.loads(trash_path.read_text(encoding="utf-8"))
        payload["deleted"] = False
        payload.pop("deleted_at", None)
        active_path = self._entry_path(entry_id)
        active_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        trash_path.unlink()
        self._commit_if_needed(f"custom-model undelete {entry_id}")
        return payload

    def list_history(self, entry_id: str) -> list[dict[str, Any]]:
        if self.git_executable is None:
            return []
        relative_path = self._relative_entry_path(entry_id)
        result = self._run_git(
            ["log", "--format=%H%x1f%cI%x1f%s", "--", relative_path],
            check=False,
        )
        if result is None or result.returncode != 0 or not result.stdout.strip():
            return []
        revisions: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            commit, committed_at, subject = line.split("\x1f")
            revisions.append(
                {
                    "commit": commit,
                    "committed_at": committed_at,
                    "subject": subject,
                }
            )
        return revisions

    def restore_entry(self, entry_id: str, revision: str) -> dict[str, Any]:
        if self.git_executable is None:
            raise RuntimeError("git backend unavailable")
        relative_path = self._relative_entry_path(entry_id)
        show_result = self._run_git(
            ["show", f"{revision}:{relative_path}"],
            check=False,
        )
        if show_result is None or show_result.returncode != 0:
            raise FileNotFoundError(f"未找到可恢复版本: {revision}:{relative_path}")
        payload = json.loads(show_result.stdout)
        path = self._entry_path(entry_id)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._commit_if_needed(f"custom-model restore {entry_id} to {revision}")
        return payload
