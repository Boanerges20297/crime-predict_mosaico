import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MINIMAL_STEPS = [
    ("prepare_data", BASE_DIR / "src" / "prepare_data.py"),
    ("run_ga", BASE_DIR / "src" / "run_ga.py"),
    ("run_stgcn_train_once", BASE_DIR / "src" / "run_stgcn_train_once.py"),
]

FULL_STEPS = [
    *MINIMAL_STEPS,
    ("run_baseline_bairros", BASE_DIR / "src" / "run_baseline_bairros.py"),
    ("run_hex_experiment", BASE_DIR / "src" / "run_hex_experiment.py"),
    ("run_orientador_comparison", BASE_DIR / "src" / "run_orientador_comparison.py"),
    ("run_skforecast_phase2_bairros", BASE_DIR / "src" / "run_skforecast_phase2_bairros.py"),
    ("run_skforecast_phase2_hexagonos", BASE_DIR / "src" / "run_skforecast_phase2_hexagonos.py"),
    ("run_skforecast_phase2_comparison", BASE_DIR / "src" / "run_skforecast_phase2_comparison.py"),
]

MINIMAL_ARTIFACTS = [
    PROCESSED_DIR / "fortaleza_crimes.csv",
    PROCESSED_DIR / "fortaleza_crimes_normalizado.csv",
    PROCESSED_DIR / "best_hex_grid.json",
    PROCESSED_DIR / "stgcn_hex_forecasts.csv",
    PROCESSED_DIR / "stgcn_hex_metadata.json",
]

FULL_ARTIFACTS = [
    *MINIMAL_ARTIFACTS,
    PROCESSED_DIR / "baseline_bairros_95_summary.json",
    PROCESSED_DIR / "experimento_hexagonos_95_summary.json",
    PROCESSED_DIR / "orientador_comparison.json",
    PROCESSED_DIR / "fase2_skforecast_bairros_95_summary.json",
    PROCESSED_DIR / "fase2_skforecast_hexagonos_95_summary.json",
    PROCESSED_DIR / "fase2_skforecast_comparison.json",
]


def run_step(step_name: str, script_path: Path) -> None:
    print(f"\n=== Executando {step_name} ===")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        check=True,
    )


def print_artifact_status(artifacts: list[Path]) -> None:
    print("\n=== Artefatos esperados ===")
    for artifact in artifacts:
        exists = artifact.exists()
        size = artifact.stat().st_size if exists else 0
        print(f"{artifact.relative_to(BASE_DIR)} | exists={exists} | bytes={size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera os artefatos necessarios para o dashboard e, opcionalmente, para os experimentos completos."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Roda tambem baseline, experimento hexagonal, comparacoes e fase 2 com skforecast.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = FULL_STEPS if args.full else MINIMAL_STEPS
    expected_artifacts = FULL_ARTIFACTS if args.full else MINIMAL_ARTIFACTS

    print(f"Python: {sys.executable}")
    print(f"Workspace: {BASE_DIR}")
    print(f"Modo: {'full' if args.full else 'minimal'}")

    for step_name, script_path in steps:
        run_step(step_name, script_path)

    print_artifact_status(expected_artifacts)
    print("\nBootstrap concluido.")


if __name__ == "__main__":
    main()
