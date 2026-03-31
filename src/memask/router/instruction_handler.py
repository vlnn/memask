import re
import sqlite3

from memask.repository.instructions import (
    create_instruction,
    deactivate_all_instructions,
    deactivate_instruction,
    list_active_instructions,
)


def handle_instruction_command(conn: sqlite3.Connection, text: str) -> dict:
    body = _extract_body(text)

    if body == "":
        return _list_instructions(conn)

    if body == "clear":
        return _clear_instructions(conn)

    remove_match = re.match(r"^remove\s+(.+)$", body, re.I)
    if remove_match:
        return _remove_instruction(conn, remove_match.group(1).strip())

    return _save_instruction(conn, body)


def _extract_body(text: str) -> str:
    match = re.match(r"^[/!]instruction\s*(.*)", text.strip(), re.I)
    if match:
        return match.group(1).strip()
    return ""


def _save_instruction(conn: sqlite3.Connection, content: str) -> dict:
    instruction = create_instruction(conn, content)
    return {
        "action": "instruction_saved",
        "data": {"instruction": instruction},
    }


def _list_instructions(conn: sqlite3.Connection) -> dict:
    instructions = list_active_instructions(conn)
    return {
        "action": "instruction_listed",
        "data": {"instructions": instructions},
    }


def _clear_instructions(conn: sqlite3.Connection) -> dict:
    count = deactivate_all_instructions(conn)
    return {
        "action": "instruction_cleared",
        "data": {"count": count},
    }


def _remove_instruction(conn: sqlite3.Connection, instruction_id: str) -> dict:
    removed = deactivate_instruction(conn, instruction_id)
    if not removed:
        return {
            "action": "instruction_not_found",
            "data": {"id": instruction_id},
        }
    return {
        "action": "instruction_removed",
        "data": {"id": instruction_id},
    }
