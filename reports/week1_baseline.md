# Reporte de semana 1 — FPSD-CNN AD–CN

Estado: **implementación preparada; ejecución científica real pendiente en Kaggle**.
<!-- RESULTS_STATUS -->

## 1. Resumen ejecutivo

Se preparó una reproducción auditable del baseline FPSD-CNN con dos salidas separadas: una reconstrucción forense de la selección sobre test publicada y un baseline metodológicamente válido con selección de checkpoint en validación interna por sujeto. No se presentan métricas inventadas ni resultados de smoke test.

## 2. Fuentes y versiones

- Artículo: DOI `10.1007/s11063-026-11850-5`.
- Código upstream: commit `45f33193c8b100ec715b8606aadaf42c08027bb8`.
- Datos: OpenNeuro `ds004504` v1.0.9; metadatos commit `63b2ec152bc3cd4487d131124f10f767ebe543bc`.
- Los hashes detallados están en `reports/audit/code_audit.md`.

## 3. Sujetos utilizados

Los metadatos contienen 36 AD y 29 CN. El hold-out publicado usa 25 AD + 18 CN en entrenamiento y 11 AD + 11 CN en test. La intersección se valida automáticamente como cero. MMSE, edad y sexo no entran al modelo.

## 4. Control de calidad

El notebook 00 genera `subjects_qc.csv` con frecuencia, canales, muestras, duración, ventanas completas, valores no finitos y amplitudes robustas. Los valores observados se incorporarán después de la ejecución Kaggle.

## 5. Discrepancias entre artículo y código

La auditoría confirmó split fijo en lugar de LOSO clásico, selección sobre test, 114 frente a 95 rasgos, 128 frente a 512 filtros, 141,704 frente a 141,732 parámetros, ventanas aproximadas de 4 s, doble preprocesamiento, CSV sin encabezado, `drop_last=True`, semillas ausentes, orden de archivos no determinista y falta de movimiento explícito a GPU.

## 6. Método de extracción

`paper95` usa 19 canales canónicos, cinco bandas, ventanas exactas de 5 s, Welch Hann (`n_fft=n_per_seg=256`, `n_overlap=10`, detrend constante), normalización espectral por ventana/canal 0.5–45 Hz y media por banda. Conserva `N×19×5` y crea una vista de 95 banda-mayor.

## 7. Arquitectura y entrenamiento

Conv1D 1→512, StarReLU, pool 2, Conv1D 512→32, StarReLU, pool 2, flatten 704, densa 128, StarReLU y salida 2. Conteo exigido: 141,732. Se usa cross-entropy sin pesos, RMSprop 0.0005, batch 64, 100 épocas, `drop_last=False` y semillas 2026–2028.

## 8. Ambiente Kaggle y GPU recibida

Pendiente. `environment.json` registrará imagen, Python, librerías, CUDA, cuDNN, nombre de GPU, CPU, RAM, disco y fecha UTC.

## 9. Tiempo de ejecución

Pendiente. `benchmark.json` registrará 10 épocas, tiempo por época, memoria pico y estimación de seis corridas completas.

## 10. Réplica forense

Pendiente de Kaggle. Se etiquetará siempre como sesgada porque el checkpoint se elige observando el test en cada época.

## 11. Baseline válido

Pendiente de Kaggle. Usa validación estratificada sólo desde los 43 sujetos de entrenamiento y evalúa los 22 sujetos de test una sola vez por semilla.

## 12. Resultados por sujeto

Pendientes: balanced accuracy, accuracy, sensibilidad AD, especificidad CN, macro-F1, ROC-AUC, PR-AUC, MCC y matriz de confusión.

## 13. Resultados por ventana

Pendientes: accuracy, sensibilidad, especificidad, F1, ROC-AUC y matriz de confusión. Se reportarán como secundarios por correlación intra-sujeto.

## 14. Comparación con el artículo

Se compararán sin ajuste post hoc los resultados contra 88.21%, 88.32% y 88.36%. Una diferencia no se “corrige” para acercarse al artículo; se explica mediante la evidencia de pipeline y evaluación.

## 15. Variación entre semillas

<!-- RESULTS_TABLES -->

Se reportarán los valores individuales, media y desviación estándar de 2026, 2027 y 2028 sólo después de completar las tres.

## 16. Limitaciones

Un único dataset y un hold-out pequeño limitan generalización. La réplica forense tiene selección sobre test. Las ventanas de un sujeto no son independientes. El orden plano impone vecindades arbitrarias a la CNN. Falta el script exacto usado por los autores para las tablas del artículo.

## 17. Instrucciones exactas de reproducción

Ejecute notebook 00 con CPU e Internet, guarde el bundle como dataset privado `fpsd-paper95-features`, añádalo al notebook 01, active GPU y use “Save & Run All”. Descargue los dos bundles resultantes. Detalles en `docs/KAGGLE_RUNBOOK.md`.

## 18. Recomendación para semana 2

Después de cerrar las tres semillas, evaluar modelos sin convolución sobre el mismo `paper95`, permutaciones preregistradas del orden de rasgos y validación cruzada agrupada por sujeto. No iniciar esas comparaciones antes de congelar este baseline.
