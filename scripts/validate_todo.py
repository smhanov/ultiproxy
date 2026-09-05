#!/usr/bin/env python3
"""Validate plans/todo.md so LLM task tracking stays deterministic and compact."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODO = ROOT / "plans" / "todo.md"
ID_RE = re.compile(r"^T\d{3}[A-Z]?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
ACTIVE_STATUSES = {"ready", "in-progress", "blocked"}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def section(lines: list[str], heading: str) -> list[str]:
    marker = f"## {heading}"
    try:
        start = lines.index(marker) + 1
    except ValueError:
        raise ValueError(f"missing section: {marker}")
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def table_rows(lines: list[str], expected_header: list[str]) -> list[list[str]]:
    table = [line for line in lines if line.startswith("|")]
    if len(table) < 2:
        raise ValueError("missing markdown table")
    header = [cell.strip() for cell in table[0].strip("|").split("|")]
    if header != expected_header:
        raise ValueError(f"bad table header: expected {expected_header}, got {header}")
    rows: list[list[str]] = []
    for raw in table[2:]:
        cells = [cell.strip() for cell in raw.strip("|").split("|")]
        if len(cells) != len(expected_header):
            raise ValueError(f"bad row column count: {raw}")
        rows.append(cells)
    return rows


def clean_code(value: str) -> str:
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def main() -> int:
    errors: list[str] = []
    if not TODO.exists():
        print(f"ERROR: missing {TODO.relative_to(ROOT)}")
        return 1

    lines = TODO.read_text(encoding="utf-8").splitlines()
    try:
        active = table_rows(
            section(lines, "Active"),
            ["ID", "Pri", "Status", "Task", "Depends on", "Plan"],
        )
        completed = table_rows(
            section(lines, "Completed"),
            ["ID", "Task", "Completed", "Ref"],
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    seen: set[str] = set()
    priorities: list[int] = []
    active_ids = {row[0] for row in active}
    completed_ids = {row[0] for row in completed}

    for task_id, priority, status, task, depends, plan_cell in active:
        if not ID_RE.fullmatch(task_id):
            fail(errors, f"active: invalid ID {task_id!r}")
        if task_id in seen:
            fail(errors, f"duplicate task ID: {task_id}")
        seen.add(task_id)

        if priority not in PRIORITY_ORDER:
            fail(errors, f"{task_id}: invalid priority {priority!r}")
        else:
            priorities.append(PRIORITY_ORDER[priority])
        if status not in ACTIVE_STATUSES:
            fail(errors, f"{task_id}: invalid active status {status!r}")
        if not task or len(task) > 100:
            fail(errors, f"{task_id}: task text must be 1-100 characters")

        plan = clean_code(plan_cell)
        if not plan.startswith("plans/") or not plan.endswith(".md"):
            fail(errors, f"{task_id}: plan must be a plans/*.md path")
        elif not (ROOT / plan).is_file():
            fail(errors, f"{task_id}: plan does not exist: {plan}")

        deps = [] if depends == "-" else [item.strip() for item in depends.split(",")]
        if status == "blocked" and not deps:
            fail(errors, f"{task_id}: blocked task must name a dependency")
        for dep in deps:
            if not ID_RE.fullmatch(dep):
                fail(errors, f"{task_id}: invalid dependency {dep!r}")
            elif dep == task_id:
                fail(errors, f"{task_id}: cannot depend on itself")
            elif dep not in active_ids and dep not in completed_ids:
                fail(errors, f"{task_id}: unknown dependency {dep}")

        unresolved = [dep for dep in deps if dep in active_ids]
        if unresolved and status != "blocked":
            fail(errors, f"{task_id}: unresolved dependencies require status blocked")
        if status == "blocked" and deps and not unresolved:
            fail(errors, f"{task_id}: all dependencies are complete; change status to ready")

    if priorities != sorted(priorities):
        fail(errors, "Active tasks must be grouped in P0 -> P1 -> P2 -> P3 order")

    for task_id, task, completed_date, ref_cell in completed:
        if not ID_RE.fullmatch(task_id):
            fail(errors, f"completed: invalid ID {task_id!r}")
        if task_id in seen:
            fail(errors, f"duplicate task ID: {task_id}")
        seen.add(task_id)
        if not task or len(task) > 100:
            fail(errors, f"{task_id}: completed task text must be 1-100 characters")
        if not DATE_RE.fullmatch(completed_date):
            fail(errors, f"{task_id}: invalid completion date {completed_date!r}")
        if not clean_code(ref_cell):
            fail(errors, f"{task_id}: completion ref is required")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"OK: {len(active)} active tasks, {len(completed)} completed task IDs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
