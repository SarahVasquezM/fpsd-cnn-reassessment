#!/usr/bin/env python
"""Replace clearly marked report placeholders using completed result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.select_dtypes(include="number").columns:
        values[column] = values[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "NA")
    headers = [str(column) for column in values.columns]
    rows = [[str(value) for value in row] for row in values.to_numpy()]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reports/week1_baseline.md"))
    args = parser.parse_args()
    aggregate = args.results / "metrics_aggregate_subject.csv"
    per_seed = args.results / "metrics_by_seed_subject.csv"
    if not aggregate.exists() or not per_seed.exists():
        raise SystemExit("Run scripts/evaluate.py after all three seeds before generating the final report")
    completed = pd.read_csv(per_seed).groupby("protocol")["seed"].apply(lambda values: set(values.astype(int)))
    required = {2026, 2027, 2028}
    protocols = {"forensic_test_selected", "valid_subject_disjoint"}
    if set(completed.index) != protocols or any(completed[name] != required for name in protocols):
        print("Reporte no materializado: faltan semillas completas; los bundles parciales siguen disponibles.")
        return
    subject_by_seed = pd.read_csv(per_seed)
    subject_aggregate = pd.read_csv(aggregate)
    window_by_seed = pd.read_csv(args.results / "metrics_by_seed_window.csv")
    window_aggregate = pd.read_csv(args.results / "metrics_aggregate_window.csv")
    text = args.report.read_text(encoding="utf-8")
    text = text.replace("<!-- RESULTS_STATUS -->", "Resultados completos disponibles para ambas configuraciones y las tres semillas.")
    text = text.replace("Pendiente. `environment.json` registrará imagen, Python, librerías, CUDA, cuDNN, nombre de GPU, CPU, RAM, disco y fecha UTC.", "Ambiente registrado en `results/metrics/environment.json`, incluyendo Python, librerías, CUDA, cuDNN, GPU, CPU, RAM, disco y fecha UTC.")
    text = text.replace("Pendiente. `benchmark.json` registrará 10 épocas, tiempo por época, memoria pico y estimación de seis corridas completas.", "Benchmark de 10 épocas registrado en `results/metrics/benchmark.json`, con tiempo por época, memoria pico y estimación de las seis corridas.")
    text = text.replace("Pendiente de Kaggle. Se etiquetará siempre como sesgada porque el checkpoint se elige observando el test en cada época.", "Ejecución completada y etiquetada como sesgada: el checkpoint se eligió observando accuracy de test en cada época. Las cifras aparecen en las tablas de las secciones 12–15.")
    text = text.replace("Pendiente de Kaggle. Usa validación estratificada sólo desde los 43 sujetos de entrenamiento y evalúa los 22 sujetos de test una sola vez por semilla.", "Ejecución completada con validación estratificada sólo desde los 43 sujetos de entrenamiento y evaluación única de los 22 sujetos de test por semilla.")
    text = text.replace("Pendientes: balanced accuracy, accuracy, sensibilidad AD, especificidad CN, macro-F1, ROC-AUC, PR-AUC, MCC y matriz de confusión.", markdown_table(subject_by_seed))
    text = text.replace("Pendientes: accuracy, sensibilidad, especificidad, F1, ROC-AUC y matriz de confusión. Se reportarán como secundarios por correlación intra-sujeto.", markdown_table(window_by_seed) + "\n\nEstas métricas son secundarias por correlación intra-sujeto.")
    text = text.replace("<!-- RESULTS_TABLES -->", "### Agregación por sujeto\n\n" + markdown_table(subject_aggregate) + "\n\n### Agregación por ventana\n\n" + markdown_table(window_aggregate))
    args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
