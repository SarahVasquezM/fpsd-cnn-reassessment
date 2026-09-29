# Reporte de semana 1 — FPSD-CNN AD–CN

Estado: **ejecución científica completada en Kaggle; tres semillas verificadas para ambos protocolos**.
Resultados completos disponibles para ambas configuraciones y las tres semillas.

## 1. Resumen ejecutivo

Se completó una reproducción auditable del baseline FPSD-CNN con dos salidas separadas. La reconstrucción forense y el baseline válido alcanzaron ambos 93.94% de balanced accuracy media por sujeto. Cada cifra resume las semillas 2026–2028 y ninguna ejecución parcial o smoke test se presenta como resultado científico.

## 2. Fuentes y versiones

- Artículo: DOI `10.1007/s11063-026-11850-5`.
- Código upstream: commit `45f33193c8b100ec715b8606aadaf42c08027bb8`.
- Datos: OpenNeuro `ds004504` v1.0.9; metadatos commit `63b2ec152bc3cd4487d131124f10f767ebe543bc`.
- Los hashes detallados están en `reports/audit/code_audit.md`.
- Código ejecutado: commit `8f8ab4e1f10be70981ce4b9f02ea0539bd4ac1e8`, registrado también en `results/metrics/github_commit.txt`.
- Ejecución final: notebook Kaggle versión 7, `scriptVersionId=353772449`, completada en 10 min 50 s.

## 3. Sujetos utilizados

Los metadatos contienen 36 AD y 29 CN. El hold-out publicado usa 25 AD + 18 CN en entrenamiento y 11 AD + 11 CN en test. La intersección se valida automáticamente como cero. MMSE, edad y sexo no entran al modelo.

## 4. Control de calidad

Los 65 sujetos aprobaron QC a 500 Hz y 19 canales. Se obtuvieron 10,615 ventanas completas de 5 s, entre 61 y 256 por sujeto, sin NaN ni infinitos. La duración mediana fue 817.64 s y el rango de amplitud p99 fue 75.35–112.92 µV. El detalle está en `results/qc/subjects_qc.csv` y el manifiesto de características en `results/qc/feature_dataset_manifest.json`.

## 5. Discrepancias entre artículo y código

La auditoría confirmó split fijo en lugar de LOSO clásico, selección sobre test, 114 frente a 95 rasgos, 128 frente a 512 filtros, 141,704 frente a 141,732 parámetros, ventanas aproximadas de 4 s, doble preprocesamiento, CSV sin encabezado, `drop_last=True`, semillas ausentes, orden de archivos no determinista y falta de movimiento explícito a GPU.

## 6. Método de extracción

`paper95` usa 19 canales canónicos, cinco bandas, ventanas exactas de 5 s, Welch Hann (`n_fft=n_per_seg=256`, `n_overlap=10`, detrend constante), normalización espectral por ventana/canal 0.5–45 Hz y media por banda. Conserva `N×19×5` y crea una vista de 95 banda-mayor.

## 7. Arquitectura y entrenamiento

Conv1D 1→512, StarReLU, pool 2, Conv1D 512→32, StarReLU, pool 2, flatten 704, densa 128, StarReLU y salida 2. Conteo exigido: 141,732. Se usa cross-entropy sin pesos, RMSprop 0.0005, batch 64, 100 épocas, `drop_last=False` y semillas 2026–2028.

## 8. Ambiente Kaggle y GPU recibida

La ejecución usó Tesla T4, Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8 y cuDNN 91002. Kaggle asignó 4 CPU lógicas y 33.66 GB de RAM. El registro completo está en `results/metrics/environment.json`.

## 9. Tiempo de ejecución

El benchmark de 10 épocas tardó 7.94 s (0.794 s/época). Proyectó 476.51 s, o 0.1324 h, para seis corridas y registró 89,718,784 bytes de memoria GPU pico; cabía holgadamente en el límite de 10 h.

## 10. Réplica forense

Ejecución completada y etiquetada como sesgada: el checkpoint se eligió observando accuracy de test en cada época. Las cifras aparecen en las tablas de las secciones 12–15.

## 11. Baseline válido

Ejecución completada con validación estratificada sólo desde los 43 sujetos de entrenamiento y evaluación única de los 22 sujetos de test por semilla.

## 12. Resultados por sujeto

| protocol | seed | accuracy | balanced_accuracy | confusion_matrix | confusion_matrix_labels | f1_AD | macro_f1 | mcc | n | pr_auc | roc_auc | sensitivity_AD | specificity_CN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forensic_test_selected | 2026.0000 | 0.9545 | 0.9545 | [[11, 0], [1, 10]] | ['CN', 'AD'] | 0.9524 | 0.9545 | 0.9129 | 22.0000 | 1.0000 | 1.0000 | 0.9091 | 1.0000 |
| forensic_test_selected | 2027.0000 | 1.0000 | 1.0000 | [[11, 0], [0, 11]] | ['CN', 'AD'] | 1.0000 | 1.0000 | 1.0000 | 22.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| forensic_test_selected | 2028.0000 | 0.8636 | 0.8636 | [[11, 0], [3, 8]] | ['CN', 'AD'] | 0.8421 | 0.8611 | 0.7559 | 22.0000 | 0.9924 | 0.9917 | 0.7273 | 1.0000 |
| valid_subject_disjoint | 2026.0000 | 0.9545 | 0.9545 | [[10, 1], [0, 11]] | ['CN', 'AD'] | 0.9565 | 0.9545 | 0.9129 | 22.0000 | 0.9842 | 0.9835 | 1.0000 | 0.9091 |
| valid_subject_disjoint | 2027.0000 | 0.9091 | 0.9091 | [[9, 2], [0, 11]] | ['CN', 'AD'] | 0.9167 | 0.9083 | 0.8321 | 22.0000 | 0.9924 | 0.9917 | 1.0000 | 0.8182 |
| valid_subject_disjoint | 2028.0000 | 0.9545 | 0.9545 | [[10, 1], [0, 11]] | ['CN', 'AD'] | 0.9565 | 0.9545 | 0.9129 | 22.0000 | 0.9924 | 0.9917 | 1.0000 | 0.9091 |

## 13. Resultados por ventana

| protocol | seed | accuracy | balanced_accuracy | confusion_matrix | confusion_matrix_labels | f1_AD | loss | macro_f1 | mcc | n | pr_auc | roc_auc | sensitivity_AD | specificity_CN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forensic_test_selected | 2026.0000 | 0.8618 | 0.8611 | [[1626, 212], [282, 1454]] | ['CN', 'AD'] | 0.8548 | 0.3194 | 0.8615 | 0.7236 | 3574.0000 | 0.9304 | 0.9392 | 0.8376 | 0.8847 |
| forensic_test_selected | 2027.0000 | 0.8677 | 0.8678 | [[1583, 255], [218, 1518]] | ['CN', 'AD'] | 0.8652 | 0.3278 | 0.8676 | 0.7354 | 3574.0000 | 0.9351 | 0.9441 | 0.8744 | 0.8613 |
| forensic_test_selected | 2028.0000 | 0.8355 | 0.8331 | [[1687, 151], [437, 1299]] | ['CN', 'AD'] | 0.8154 | 0.3931 | 0.8335 | 0.6780 | 3574.0000 | 0.9251 | 0.9274 | 0.7483 | 0.9178 |
| valid_subject_disjoint | 2026.0000 | 0.8330 | 0.8347 | [[1424, 414], [183, 1553]] | ['CN', 'AD'] | 0.8388 | 0.5140 | 0.8327 | 0.6725 | 3574.0000 | 0.8952 | 0.9113 | 0.8946 | 0.7748 |
| valid_subject_disjoint | 2027.0000 | 0.8355 | 0.8382 | [[1365, 473], [115, 1621]] | ['CN', 'AD'] | 0.8465 | 0.4843 | 0.8346 | 0.6863 | 3574.0000 | 0.8980 | 0.9138 | 0.9338 | 0.7427 |
| valid_subject_disjoint | 2028.0000 | 0.8372 | 0.8390 | [[1425, 413], [169, 1567]] | ['CN', 'AD'] | 0.8434 | 0.5079 | 0.8369 | 0.6817 | 3574.0000 | 0.8943 | 0.9111 | 0.9026 | 0.7753 |

Estas métricas son secundarias por correlación intra-sujeto.

## 14. Comparación con el artículo

El artículo informa 88.21%, 88.32% y 88.36%. El baseline válido obtuvo 93.94% ± 2.62% de balanced accuracy por sujeto, una diferencia descriptiva de +5.73, +5.62 y +5.58 puntos porcentuales. No se interpreta como una mejora directa: cambian la unidad de evaluación, la selección de checkpoint y varios detalles del pipeline, y el hold-out sólo contiene 22 sujetos.

## 15. Variación entre semillas

### Agregación por sujeto

| protocol | accuracy_mean | accuracy_std | balanced_accuracy_mean | balanced_accuracy_std | f1_AD_mean | f1_AD_std | macro_f1_mean | macro_f1_std | mcc_mean | mcc_std | n_mean | n_std | pr_auc_mean | pr_auc_std | roc_auc_mean | roc_auc_std | sensitivity_AD_mean | sensitivity_AD_std | specificity_CN_mean | specificity_CN_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forensic_test_selected | 0.9394 | 0.0694 | 0.9394 | 0.0694 | 0.9315 | 0.0810 | 0.9385 | 0.0708 | 0.8896 | 0.1237 | 22.0000 | 0.0000 | 0.9975 | 0.0044 | 0.9972 | 0.0048 | 0.8788 | 0.1389 | 1.0000 | 0.0000 |
| valid_subject_disjoint | 0.9394 | 0.0262 | 0.9394 | 0.0262 | 0.9432 | 0.0230 | 0.9391 | 0.0266 | 0.8859 | 0.0467 | 22.0000 | 0.0000 | 0.9897 | 0.0048 | 0.9890 | 0.0048 | 1.0000 | 0.0000 | 0.8788 | 0.0525 |

### Agregación por ventana

| protocol | accuracy_mean | accuracy_std | balanced_accuracy_mean | balanced_accuracy_std | f1_AD_mean | f1_AD_std | loss_mean | loss_std | macro_f1_mean | macro_f1_std | mcc_mean | mcc_std | n_mean | n_std | pr_auc_mean | pr_auc_std | roc_auc_mean | roc_auc_std | sensitivity_AD_mean | sensitivity_AD_std | specificity_CN_mean | specificity_CN_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forensic_test_selected | 0.8550 | 0.0171 | 0.8540 | 0.0184 | 0.8451 | 0.0262 | 0.3468 | 0.0403 | 0.8542 | 0.0182 | 0.7123 | 0.0303 | 3574.0000 | 0.0000 | 0.9302 | 0.0050 | 0.9369 | 0.0086 | 0.8201 | 0.0649 | 0.8879 | 0.0284 |
| valid_subject_disjoint | 0.8352 | 0.0021 | 0.8373 | 0.0023 | 0.8429 | 0.0039 | 0.5021 | 0.0157 | 0.8348 | 0.0021 | 0.6802 | 0.0070 | 3574.0000 | 0.0000 | 0.8958 | 0.0019 | 0.9121 | 0.0015 | 0.9103 | 0.0207 | 0.7642 | 0.0187 |

Las tablas anteriores incluyen los valores individuales y la media ± desviación estándar de las tres semillas completas: 2026, 2027 y 2028.

## 16. Limitaciones

Un único dataset y un hold-out pequeño limitan generalización. La réplica forense tiene selección sobre test. Las ventanas de un sujeto no son independientes. El orden plano impone vecindades arbitrarias a la CNN. Falta el script exacto usado por los autores para las tablas del artículo.

## 17. Instrucciones exactas de reproducción

Ejecute notebook 00 con CPU e Internet, guarde el bundle como dataset privado `fpsd-paper95-features`, añádalo al notebook 01, active GPU y use “Save & Run All”. Descargue los dos bundles resultantes. Detalles en `docs/KAGGLE_RUNBOOK.md`.

## 18. Recomendación para semana 2

Después de cerrar las tres semillas, evaluar modelos sin convolución sobre el mismo `paper95`, permutaciones preregistradas del orden de rasgos y validación cruzada agrupada por sujeto. No iniciar esas comparaciones antes de congelar este baseline.
