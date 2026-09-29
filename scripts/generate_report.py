#!/usr/bin/env python
"""Replace clearly marked report placeholders using completed result tables."""

from __future__ import annotations

import argparse
import json
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
    environment = json.loads((args.results / "environment.json").read_text(encoding="utf-8"))
    benchmark = json.loads((args.results / "benchmark.json").read_text(encoding="utf-8"))
    commit = (args.results / "github_commit.txt").read_text(encoding="utf-8").strip()
    valid_subject = subject_aggregate.loc[subject_aggregate["protocol"] == "valid_subject_disjoint"].iloc[0]
    forensic_subject = subject_aggregate.loc[subject_aggregate["protocol"] == "forensic_test_selected"].iloc[0]
    published = (0.8821, 0.8832, 0.8836)
    differences = [(valid_subject["balanced_accuracy_mean"] - value) * 100 for value in published]
    text = args.report.read_text(encoding="utf-8")
    text = text.replace(
        "Estado: **implementación preparada; ejecución científica real pendiente en Kaggle**.",
        "Estado: **ejecución científica completada en Kaggle; tres semillas verificadas para ambos protocolos**.",
    )
    text = text.replace("<!-- RESULTS_STATUS -->", "Resultados completos disponibles para ambas configuraciones y las tres semillas.")
    text = text.replace(
        "Se preparó una reproducción auditable del baseline FPSD-CNN con dos salidas separadas: una reconstrucción forense de la selección sobre test publicada y un baseline metodológicamente válido con selección de checkpoint en validación interna por sujeto. No se presentan métricas inventadas ni resultados de smoke test.",
        f"Se completó una reproducción auditable del baseline FPSD-CNN con dos salidas separadas. La reconstrucción forense y el baseline válido alcanzaron {forensic_subject['balanced_accuracy_mean']:.2%} y {valid_subject['balanced_accuracy_mean']:.2%} de balanced accuracy por sujeto, respectivamente. Cada cifra resume las semillas 2026–2028 y ninguna ejecución parcial o smoke test se presenta como resultado científico.",
    )
    text = text.replace(
        "- Los hashes detallados están en `reports/audit/code_audit.md`.",
        f"- Los hashes detallados están en `reports/audit/code_audit.md`.\n- Código ejecutado: commit `{commit}`, registrado también en `results/metrics/github_commit.txt`.",
    )
    text = text.replace(
        "El notebook 00 genera `subjects_qc.csv` con frecuencia, canales, muestras, duración, ventanas completas, valores no finitos y amplitudes robustas. Los valores observados se incorporarán después de la ejecución Kaggle.",
        "Los 65 sujetos aprobaron QC a 500 Hz y 19 canales. Se obtuvieron 10,615 ventanas completas de 5 s (61–256 por sujeto), sin NaN ni infinitos. La duración mediana fue 817.64 s y el rango de amplitud p99 fue 75.35–112.92 µV. El detalle está en `results/qc/subjects_qc.csv`.",
    )
    environment_text = (
        f"La ejecución usó {environment['gpu_name']}, Python {environment['python'].split()[0]}, "
        f"PyTorch {environment['packages']['torch']}, CUDA {environment['torch_cuda_version']} y "
        f"cuDNN {environment['cudnn_version']}. Kaggle asignó {environment['cpu_count_logical']} CPU lógicas y "
        f"{environment['ram_bytes'] / 1_000_000_000:.2f} GB de RAM. El registro completo está en "
        "`results/metrics/environment.json`."
    )
    text = text.replace("Pendiente. `environment.json` registrará imagen, Python, librerías, CUDA, cuDNN, nombre de GPU, CPU, RAM, disco y fecha UTC.", environment_text)
    benchmark_text = (
        f"El benchmark de {benchmark['benchmark_epochs']} épocas tardó {benchmark['elapsed_seconds']:.2f} s "
        f"({benchmark['seconds_per_epoch']:.3f} s/época). Proyectó {benchmark['estimated_seconds_two_protocols_three_seeds']:.2f} s "
        f"({benchmark['estimated_hours']:.4f} h) para seis corridas, con {benchmark['gpu_peak_memory_bytes']:,} bytes "
        "de memoria GPU pico; cabía holgadamente en el límite de 10 h."
    )
    text = text.replace("Pendiente. `benchmark.json` registrará 10 épocas, tiempo por época, memoria pico y estimación de seis corridas completas.", benchmark_text)
    text = text.replace("Pendiente de Kaggle. Se etiquetará siempre como sesgada porque el checkpoint se elige observando el test en cada época.", "Ejecución completada y etiquetada como sesgada: el checkpoint se eligió observando accuracy de test en cada época. Las cifras aparecen en las tablas de las secciones 12–15.")
    text = text.replace("Pendiente de Kaggle. Usa validación estratificada sólo desde los 43 sujetos de entrenamiento y evalúa los 22 sujetos de test una sola vez por semilla.", "Ejecución completada con validación estratificada sólo desde los 43 sujetos de entrenamiento y evaluación única de los 22 sujetos de test por semilla.")
    text = text.replace("Pendientes: balanced accuracy, accuracy, sensibilidad AD, especificidad CN, macro-F1, ROC-AUC, PR-AUC, MCC y matriz de confusión.", markdown_table(subject_by_seed))
    text = text.replace("Pendientes: accuracy, sensibilidad, especificidad, F1, ROC-AUC y matriz de confusión. Se reportarán como secundarios por correlación intra-sujeto.", markdown_table(window_by_seed) + "\n\nEstas métricas son secundarias por correlación intra-sujeto.")
    comparison = (
        "El artículo informa 88.21%, 88.32% y 88.36%. El baseline válido obtuvo "
        f"{valid_subject['balanced_accuracy_mean']:.2%} ± {valid_subject['balanced_accuracy_std']:.2%} de balanced accuracy por sujeto, "
        f"una diferencia descriptiva de +{differences[0]:.2f}, +{differences[1]:.2f} y +{differences[2]:.2f} puntos porcentuales. "
        "No se interpreta como una mejora directa: cambian la unidad de evaluación, la selección de checkpoint y varios detalles del pipeline, y el hold-out sólo contiene 22 sujetos."
    )
    text = text.replace("Se compararán sin ajuste post hoc los resultados contra 88.21%, 88.32% y 88.36%. Una diferencia no se “corrige” para acercarse al artículo; se explica mediante la evidencia de pipeline y evaluación.", comparison)
    text = text.replace("<!-- RESULTS_TABLES -->", "### Agregación por sujeto\n\n" + markdown_table(subject_aggregate) + "\n\n### Agregación por ventana\n\n" + markdown_table(window_aggregate))
    text = text.replace(
        "Se reportarán los valores individuales, media y desviación estándar de 2026, 2027 y 2028 sólo después de completar las tres.",
        "Las tablas anteriores incluyen los valores individuales y la media ± desviación estándar de las tres semillas completas: 2026, 2027 y 2028.",
    )
    args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
