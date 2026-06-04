import os
import sys
from datetime import datetime

from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dashboard_service import (  # noqa: E402
    DEFAULT_GENERATIONS,
    DEFAULT_POP_SIZE,
    analyze_filters,
    get_available_bairros,
    load_base_dataframe,
)


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
BASE_DF = load_base_dataframe()
AVAILABLE_BAIRROS = get_available_bairros(BASE_DF)


def read_filter_args():
    selected_bairros = request.args.getlist("bairro")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    pop_size = int(request.args.get("pop_size", DEFAULT_POP_SIZE))
    generations = int(request.args.get("generations", DEFAULT_GENERATIONS))
    hide_sparse_hexes = request.args.get("hide_sparse_hexes") == "1"
    show_cvli_points = request.args.get("show_cvli_points") == "1"
    return selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points


def build_analysis_from_request():
    selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points = read_filter_args()

    result = analyze_filters(
        BASE_DF,
        selected_bairros=selected_bairros,
        start_date=start_date or None,
        end_date=end_date or None,
        pop_size=pop_size,
        generations=generations,
        hide_sparse_hexes=hide_sparse_hexes,
        show_cvli_points=show_cvli_points,
    )
    has_active_filters = bool(selected_bairros or start_date or end_date)
    return result, selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, has_active_filters


@app.route("/", methods=["GET"])
def index():
    result, selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, has_active_filters = build_analysis_from_request()

    if request.args.get("partial") == "1":
        return jsonify(
            {
                "map_html": result.map_html,
                "error": result.error,
                "from_cache": result.from_cache,
                "metrics": result.metrics,
            }
        )

    return render_template(
        "index.html",
        result=result,
        available_bairros=AVAILABLE_BAIRROS,
        total_bairros=len(AVAILABLE_BAIRROS),
        selected_bairros=selected_bairros,
        has_active_filters=has_active_filters,
        hide_sparse_hexes=hide_sparse_hexes,
        show_cvli_points=show_cvli_points,
        pop_size=pop_size,
        generations=generations,
        start_date=start_date,
        end_date=end_date,
    )


@app.route("/relatorio", methods=["GET"])
def report():
    result, selected_bairros, start_date, end_date, pop_size, generations, hide_sparse_hexes, show_cvli_points, has_active_filters = build_analysis_from_request()

    return render_template(
        "report.html",
        result=result,
        selected_bairros=selected_bairros,
        total_bairros=len(AVAILABLE_BAIRROS),
        has_active_filters=has_active_filters,
        hide_sparse_hexes=hide_sparse_hexes,
        show_cvli_points=show_cvli_points,
        pop_size=pop_size,
        generations=generations,
        start_date=start_date,
        end_date=end_date,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
