import json
import os
import shutil
import sys
from datetime import datetime
from functools import wraps

from flask import Flask, abort, jsonify, render_template, request, send_file, session
from werkzeug.utils import secure_filename


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Load .env without extra dependency
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from dashboard_service import (  # noqa: E402
    DEFAULT_GENERATIONS,
    DEFAULT_POP_SIZE,
    analyze_filters,
    get_available_bairros,
    load_base_dataframe,
)


ORIENTADOR_PASSWORD = os.environ.get("ORIENTADOR_PASSWORD", "orientador@2024")
ORIENTADOR_DIR = os.path.join(BASE_DIR, "data", "orientador")
ENTREGAS_DIR = os.path.join(ORIENTADOR_DIR, "entregas")
CHAT_FILE = os.path.join(ORIENTADOR_DIR, "chat.json")

os.makedirs(ENTREGAS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "crime-predict-fallback-secret")

BASE_DF = load_base_dataframe()
AVAILABLE_BAIRROS = get_available_bairros(BASE_DF)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ensure_chat():
    if not os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def _safe_path(rel):
    """Resolve rel relative to ENTREGAS_DIR, blocking path traversal."""
    base = os.path.realpath(ENTREGAS_DIR)
    if not rel:
        return base
    target = os.path.realpath(os.path.join(base, rel))
    if target == base or target.startswith(base + os.sep):
        return target
    return None


def orientador_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("orientador_auth"):
            return jsonify({"error": "Não autorizado"}), 401
        return f(*args, **kwargs)
    return decorated


# ─── Dashboard routes ─────────────────────────────────────────────────────────

def read_filter_args():
    selected_bairros = request.args.getlist("bairro")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    pop_size = int(request.args.get("pop_size", DEFAULT_POP_SIZE))
    generations = int(request.args.get("generations", DEFAULT_GENERATIONS))
    hide_sparse_hexes = request.args.get("hide_sparse_hexes") == "1"
    show_cvli_points = request.args.get("show_cvli_points") == "1"
    show_bairro_heatmap = request.args.get("show_bairro_heatmap") == "1"
    return selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, show_bairro_heatmap


def build_analysis_from_request():
    selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, show_bairro_heatmap = read_filter_args()
    result = analyze_filters(
        BASE_DF,
        selected_bairros=selected_bairros,
        start_date=start_date or None,
        end_date=end_date or None,
        pop_size=pop_size,
        generations=generations,
        hide_sparse_hexes=hide_sparse_hexes,
        show_cvli_points=show_cvli_points,
        show_bairro_heatmap=show_bairro_heatmap,
    )
    has_active_filters = bool(selected_bairros or start_date or end_date)
    return result, selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, show_bairro_heatmap, has_active_filters


@app.route("/", methods=["GET"])
def index():
    result, selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, show_bairro_heatmap, has_active_filters = build_analysis_from_request()

    if request.args.get("partial") == "1":
        return jsonify({
            "map_html": result.map_html,
            "error": result.error,
            "from_cache": result.from_cache,
            "metrics": result.metrics,
        })

    return render_template(
        "index.html",
        result=result,
        available_bairros=AVAILABLE_BAIRROS,
        total_bairros=len(AVAILABLE_BAIRROS),
        selected_bairros=selected_bairros,
        has_active_filters=has_active_filters,
        hide_sparse_hexes=hide_sparse_hexes,
        show_cvli_points=show_cvli_points,
        show_bairro_heatmap=show_bairro_heatmap,
        pop_size=pop_size,
        generations=generations,
        start_date=start_date,
        end_date=end_date,
    )


@app.route("/relatorio", methods=["GET"])
def report():
    result, selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, show_bairro_heatmap, has_active_filters = build_analysis_from_request()

    return render_template(
        "report.html",
        result=result,
        selected_bairros=selected_bairros,
        total_bairros=len(AVAILABLE_BAIRROS),
        has_active_filters=has_active_filters,
        hide_sparse_hexes=hide_sparse_hexes,
        show_cvli_points=show_cvli_points,
        show_bairro_heatmap=show_bairro_heatmap,
        pop_size=pop_size,
        generations=generations,
        start_date=start_date,
        end_date=end_date,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


# ─── Orientador routes ────────────────────────────────────────────────────────

@app.route("/orientador/login", methods=["POST"])
def orientador_login():
    data = request.get_json(silent=True) or {}
    if data.get("password") == ORIENTADOR_PASSWORD:
        session["orientador_auth"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Senha incorreta"}), 401


@app.route("/orientador/logout", methods=["POST"])
def orientador_logout():
    session.pop("orientador_auth", None)
    return jsonify({"ok": True})


@app.route("/orientador/api/files")
@orientador_required
def orientador_files():
    rel = request.args.get("path", "")
    safe = _safe_path(rel)
    if safe is None or not os.path.exists(safe):
        return jsonify({"error": "Caminho inválido"}), 400

    items = []
    for entry in sorted(os.scandir(safe), key=lambda e: (not e.is_dir(), e.name.lower())):
        stat = entry.stat()
        item_rel = (rel.rstrip("/") + "/" + entry.name).lstrip("/") if rel else entry.name
        items.append({
            "name": entry.name,
            "is_dir": entry.is_dir(),
            "size": stat.st_size if not entry.is_dir() else None,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y %H:%M"),
            "path": item_rel,
        })

    return jsonify({"items": items, "path": rel})


@app.route("/orientador/api/upload", methods=["POST"])
@orientador_required
def orientador_upload():
    rel = request.form.get("path", "")
    safe_dir = _safe_path(rel)
    if safe_dir is None:
        return jsonify({"error": "Caminho inválido"}), 400

    uploaded = []
    for f in request.files.getlist("files"):
        fname = secure_filename(f.filename)
        if fname:
            f.save(os.path.join(safe_dir, fname))
            uploaded.append(fname)

    return jsonify({"ok": True, "uploaded": uploaded})


@app.route("/orientador/api/mkdir", methods=["POST"])
@orientador_required
def orientador_mkdir():
    data = request.get_json(silent=True) or {}
    safe = _safe_path(data.get("path", ""))
    if safe is None:
        return jsonify({"error": "Caminho inválido"}), 400
    os.makedirs(safe, exist_ok=True)
    return jsonify({"ok": True})


@app.route("/orientador/api/delete", methods=["POST"])
@orientador_required
def orientador_delete():
    data = request.get_json(silent=True) or {}
    safe = _safe_path(data.get("path", ""))
    if safe is None or not os.path.exists(safe):
        return jsonify({"error": "Não encontrado"}), 404
    if os.path.isdir(safe):
        shutil.rmtree(safe)
    else:
        os.unlink(safe)
    return jsonify({"ok": True})


@app.route("/orientador/api/download")
@orientador_required
def orientador_download():
    safe = _safe_path(request.args.get("path", ""))
    if safe is None or not os.path.isfile(safe):
        abort(404)
    return send_file(safe, as_attachment=True, download_name=os.path.basename(safe))


@app.route("/orientador/api/chat", methods=["GET"])
@orientador_required
def orientador_chat_get():
    _ensure_chat()
    with open(CHAT_FILE, encoding="utf-8") as f:
        messages = json.load(f)
    return jsonify({"messages": messages})


@app.route("/orientador/api/chat", methods=["POST"])
@orientador_required
def orientador_chat_post():
    _ensure_chat()
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    author = data.get("author", "aluno")
    if not text:
        return jsonify({"error": "Mensagem vazia"}), 400
    if author not in ("orientador", "aluno"):
        author = "aluno"

    with open(CHAT_FILE, encoding="utf-8") as f:
        messages = json.load(f)

    msg = {
        "id": len(messages) + 1,
        "author": author,
        "text": text,
        "timestamp": datetime.now().isoformat(),
    }
    messages.append(msg)

    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True, "message": msg})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
