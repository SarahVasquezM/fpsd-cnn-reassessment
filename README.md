# FPSD-CNN reassessment

Reproducción auditable del baseline FPSD-CNN para clasificación AD–CN y CN–FTD con los derivados preprocesados de OpenNeuro `ds004504` versión `1.0.9`.

El repositorio separa dos preguntas:

- `forensic_test_selected`: reconstruye la selección de checkpoint con el test observada en el código publicado. Es una estimación sesgada y nunca se presenta como evaluación limpia.
- `valid_subject_disjoint`: crea validación sólo desde los 43 sujetos de entrenamiento, selecciona el checkpoint allí y evalúa una sola vez los 22 sujetos del test publicado.

La ejecución científica ocurre exclusivamente en Kaggle. Windows se usa para Codex, navegador y revisión opcional; no se entrena localmente.

La semana 2 añade un bundle de 88 sujetos, el holdout CN–FTD publicado, validación agrupada de cinco folds, balanceo reproducible sólo en entrenamiento y LOSO verdadero. La guía paso a paso está en [`docs/WEEK2_RUNBOOK.md`](docs/WEEK2_RUNBOOK.md) y la especificación científica en `reports/week2_grouped_validation.md`.

## Estado verificado

La preparación y el entrenamiento final están completos:

- 65 sujetos AD–CN: 36 AD y 29 CN; los 65 aprobaron QC.
- 10,615 ventanas de 5 s; tensor `10,615 × 19 × 5` y vista plana de 95 rasgos.
- Tres semillas completas (`2026`, `2027`, `2028`) para ambos protocolos.
- 12 pruebas aprobadas antes del entrenamiento; GPU Tesla T4; 141,732 parámetros.
- Balanced accuracy media por sujeto: `93.94% ± 6.94%` forense y `93.94% ± 2.62%` válida.
- Commit registrado en los resultados: `8f8ab4e1f10be70981ce4b9f02ea0539bd4ac1e8`.

La ejecución final está en [Kaggle, versión 7](https://www.kaggle.com/code/sarahvasquez97/01-train-baseline-kaggle?scriptVersionId=353772449). El reporte completo está en `reports/week1_baseline.md` y las tablas verificadas en `results/metrics/`.

## Evidencia congelada

- Artículo: Li et al., *Neural Processing Letters* (2026), DOI `10.1007/s11063-026-11850-5`.
- Upstream: `changjingzhi/FPSD-CNN` commit `45f33193c8b100ec715b8606aadaf42c08027bb8`.
- Dataset: OpenNeuro `ds004504` v`1.0.9`, commit de metadatos `63b2ec152bc3cd4487d131124f10f767ebe543bc`.
- Split fijo: 25 AD + 18 CN para entrenamiento; 11 AD + 11 CN para test; intersección cero.

No se redistribuye el upstream ni los datos. Sus hashes, atribución y discrepancias se conservan en `reports/audit/`.

## Ejecución

1. Importe `notebooks/00_prepare_features_kaggle.ipynb` en un notebook Kaggle con CPU y adjunte el mirror `thngdngvn/openneuro-ds004504` si Internet no está disponible.
2. Guarde `fpsd_features_bundle.zip` como dataset privado `fpsd-paper95-features`.
3. Importe `notebooks/01_train_baseline_kaggle.ipynb`, añada `fpsd-paper95-features`, el paquete privado de código y active GPU.
4. Ejecute “Save & Run All”. El notebook verifica hashes, corre las pruebas, hace el benchmark y sólo después entrena.
5. Descargue `github_results_bundle.zip` e incorpórelo siguiendo `docs/KAGGLE_RUNBOOK.md`.

Para semana 2, use `02_prepare_week2_features_kaggle.ipynb` y después `03_train_week2_kaggle.ipynb`. El segundo notebook comienza con un benchmark LOSO de cinco folds y permite reanudar rangos de folds sin mezclar configuraciones.

## Reglas científicas

- Unidad de separación: sujeto.
- Características maestras: `N × 19 × 5`; vista CNN: 95 valores en orden banda-mayor.
- Ventanas: exactamente 5 s, sin solapamiento.
- No se vuelven a filtrar, rereferenciar ni limpiar los derivados.
- Métrica primaria: balanced accuracy por sujeto; las métricas por ventana son secundarias.
- Ningún resultado se incluye hasta completar tres semillas o documentar un bloqueo cuantificado.

## Pruebas rápidas

En Kaggle:

```bash
python -m pip install -e .
pytest -q
python -c "from fpsd_reassessment.models import validate_architecture; print(validate_architecture())"
```

El último comando debe imprimir `141732`. Un valor distinto detiene el entrenamiento completo.
