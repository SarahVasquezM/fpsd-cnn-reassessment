# FPSD-CNN reassessment

Reproducción auditable del baseline FPSD-CNN para clasificación AD–CN con los derivados preprocesados de OpenNeuro `ds004504` versión `1.0.9`.

El repositorio separa dos preguntas:

- `forensic_test_selected`: reconstruye la selección de checkpoint con el test observada en el código publicado. Es una estimación sesgada y nunca se presenta como evaluación limpia.
- `valid_subject_disjoint`: crea validación sólo desde los 43 sujetos de entrenamiento, selecciona el checkpoint allí y evalúa una sola vez los 22 sujetos del test publicado.

La ejecución científica ocurre exclusivamente en Kaggle. Windows se usa para Codex, navegador y revisión opcional; no se entrena localmente.

## Evidencia congelada

- Artículo: Li et al., *Neural Processing Letters* (2026), DOI `10.1007/s11063-026-11850-5`.
- Upstream: `changjingzhi/FPSD-CNN` commit `45f33193c8b100ec715b8606aadaf42c08027bb8`.
- Dataset: OpenNeuro `ds004504` v`1.0.9`, commit de metadatos `63b2ec152bc3cd4487d131124f10f767ebe543bc`.
- Split fijo: 25 AD + 18 CN para entrenamiento; 11 AD + 11 CN para test; intersección cero.

No se redistribuye el upstream ni los datos. Sus hashes, atribución y discrepancias se conservan en `reports/audit/`.

## Ejecución

1. Importe `notebooks/00_prepare_features_kaggle.ipynb` en un notebook Kaggle con Internet y CPU.
2. Guarde `fpsd_features_bundle.zip` como dataset privado `fpsd-paper95-features`.
3. Importe `notebooks/01_train_baseline_kaggle.ipynb`, añada ese dataset privado y active GPU.
4. Ejecute “Save & Run All”. El notebook verifica hashes, corre las pruebas, hace el benchmark y sólo después entrena.
5. Descargue `github_results_bundle.zip` e incorpórelo siguiendo `docs/KAGGLE_RUNBOOK.md`.

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
