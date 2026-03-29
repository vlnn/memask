from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

if TYPE_CHECKING:
    from memask.app import AppContext


def create_app(app_context: AppContext) -> Flask:
    app = Flask(__name__)
    app.config["app_context"] = app_context

    app.add_url_rule("/health", view_func=_health, methods=["GET"])
    app.add_url_rule("/input", view_func=_input, methods=["POST"])
    app.add_url_rule("/items", view_func=_items, methods=["GET"])
    app.add_url_rule("/search", view_func=_search, methods=["GET"])
    app.add_url_rule("/settings", view_func=_settings, methods=["GET"])

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


def _items():
    from memask.repository.items import list_items

    svc = _get_svc()
    type_filter = request.args.get("type")
    status_filter = request.args.get("status")

    items = list_items(svc.conn, type=type_filter, status=status_filter)
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


def _settings():
    return jsonify({})


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
