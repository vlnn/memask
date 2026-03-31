from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

if TYPE_CHECKING:
    from memask.app import AppContext

logger = logging.getLogger(__name__)


def create_app(app_context: AppContext) -> Flask:
    app = Flask(__name__)
    app.config["app_context"] = app_context

    @app.after_request
    def cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS"
        return response

    @app.errorhandler(Exception)
    def handle_exception(exc):
        logger.exception("unhandled error: %s", exc)
        return jsonify({
            "action": "error",
            "data": {"message": str(exc)},
        }), 500

    app.add_url_rule("/health", view_func=_health, methods=["GET"])
    app.add_url_rule("/input", view_func=_input, methods=["POST"])
    app.add_url_rule("/items", view_func=_items, methods=["GET"])
    app.add_url_rule("/items/<item_id>", view_func=_patch_item, methods=["PATCH"])
    app.add_url_rule("/items/<item_id>", view_func=_delete_item, methods=["DELETE"])
    app.add_url_rule("/search", view_func=_search, methods=["GET"])
    app.add_url_rule("/suggest", view_func=_suggest, methods=["GET"])
    app.add_url_rule("/settings", view_func=_settings, methods=["GET"])
    app.add_url_rule("/instructions", view_func=_get_instructions, methods=["GET"])
    app.add_url_rule(
        "/instructions/<instruction_id>",
        view_func=_delete_instruction,
        methods=["DELETE"],
    )

    return app


def _get_svc():
    from flask import current_app
    return current_app.config["app_context"].service_context()


def _health():
    from memask.repository.jobs import queue_status

    svc = _get_svc()

    llm_info = {"available": False}
    if svc.llm is not None:
        llm_info["available"] = svc.llm.is_available()

    embedder_info = {"model": None}
    if svc.embedder is not None:
        embedder_info["model"] = svc.embedder.model_name

    jobs = queue_status(svc.conn)
    index_count = svc.store.count() if svc.store else 0

    return jsonify({
        "status": "ok",
        "llm": llm_info,
        "embedder": embedder_info,
        "jobs": jobs,
        "index": {"vectors": index_count},
    })


def _input():
    body = request.get_json(silent=True)
    if not body or not body.get("text", "").strip():
        return jsonify({"error": "text is required"}), 400

    from memask.router.dispatcher import dispatch

    svc = _get_svc()
    result = dispatch(svc, body["text"])
    return jsonify(result.to_dict())


def _parse_limit(raw, default=100, ceiling=500):
    try:
        return min(int(raw), ceiling)
    except (TypeError, ValueError):
        return default


def _items():
    from memask.repository.items import list_items

    svc = _get_svc()
    type_filter = request.args.get("type")
    status_filter = request.args.get("status")
    limit = _parse_limit(request.args.get("limit"), default=100, ceiling=500)

    items = list_items(svc.conn, type=type_filter, status=status_filter, limit=limit)
    return jsonify({
        "items": [_serialize_item(item) for item in items],
    })


def _search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "q parameter is required"}), 400

    from memask.search.keyword import keyword_search

    svc = _get_svc()
    results = keyword_search(svc.conn, query)
    return jsonify({
        "results": [
            {
                "id": r.item.id,
                "content": r.item.content,
                "type": r.item.type,
                "score": r.score,
            }
            for r in results
        ],
    })


def _suggest():
    from memask.suggest import suggest

    query = request.args.get("q", "")
    limit = request.args.get("limit", "10")

    try:
        limit_int = min(int(limit), 20)
    except ValueError:
        limit_int = 10

    svc = _get_svc()
    results = suggest(svc.conn, query, limit=limit_int)
    return jsonify({"suggestions": results})


def _patch_item(item_id):
    from memask.repository.items import VALID_TYPES, get_item, update_item

    svc = _get_svc()

    existing = get_item(svc.conn, item_id)
    if not existing or existing.deleted_at is not None:
        return jsonify({"error": "item not found"}), 404

    body = request.get_json(silent=True) or {}

    if "type" in body and body["type"] not in VALID_TYPES:
        return jsonify({"error": f"invalid type: {body['type']}"}), 400

    fields = {}
    for key in ("content", "type", "status", "title", "category", "tags", "priority"):
        if key in body:
            fields[key] = body[key]

    updated = update_item(svc.conn, item_id, **fields)
    return jsonify({"item": _serialize_item(updated)})


def _delete_item(item_id):
    from memask.repository.items import soft_delete_item

    svc = _get_svc()
    deleted = soft_delete_item(svc.conn, item_id)
    if not deleted:
        return jsonify({"error": "item not found"}), 404
    return jsonify({"deleted": True, "id": item_id})


def _settings():
    return jsonify({})


def _get_instructions():
    from memask.repository.instructions import list_active_instructions

    svc = _get_svc()
    instructions = list_active_instructions(svc.conn)
    return jsonify(instructions)


def _delete_instruction(instruction_id):
    from memask.repository.instructions import deactivate_instruction

    svc = _get_svc()
    removed = deactivate_instruction(svc.conn, instruction_id)
    if not removed:
        return jsonify({"error": "not found"}), 404
    return jsonify({"status": "deactivated", "id": instruction_id})


def _serialize_item(item):
    return {
        "id": item.id,
        "type": item.type,
        "content": item.content,
        "title": item.title,
        "status": item.status,
        "category": item.category,
        "tags": item.tags,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }
