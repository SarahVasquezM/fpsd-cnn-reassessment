# Registro de decisiones

No se corrigen contradicciones de forma silenciosa. Cada decisión se toma con la prioridad: datos/metadatos observados → código/checkpoint → artículo → README → inferencia explícita.

## 2026-09-26

- **DEC-001 — Alcance AD–CN.** Semana 1 incluye 36 AD y 29 CN; FTD se excluye.
- **DEC-002 — Snapshot.** Se fija OpenNeuro `ds004504` v1.0.9, commit de metadatos `63b2ec152bc3cd4487d131124f10f767ebe543bc`.
- **DEC-003 — Split.** Se reutilizan las listas publicadas (43 train, 22 test) y se describe como hold-out fijo por sujeto, no LOSO.
- **DEC-004 — Reparación 114→95.** Se adoptan las cinco bandas del artículo. Es una reparación documentada porque el extractor publicado de seis bandas es incompatible con su propia siguiente etapa.
- **DEC-005 — Ventanas.** Se usan intervalos exactos no solapados de 5 s y se descarta únicamente el remanente final incompleto.
- **DEC-006 — Derivados.** No se vuelve a filtrar, rereferenciar, remuestrear ni ejecutar ICA/ICLabel. Una frecuencia distinta de 500 Hz detiene `paper95`.
- **DEC-007 — Welch completado.** Además de Hann, `n_fft=256` y `n_overlap=10`, se fija `n_per_seg=256` y detrend `constant`; quedan serializados en la configuración efectiva.
- **DEC-008 — Normalización.** Cada ventana/canal se divide por la suma de bins PSD 0.5–45 Hz antes de promediar los bins de cada banda.
- **DEC-009 — Orden.** La matriz maestra es `[ventana, canal, banda]`; la vista de 95 es banda-mayor para reflejar el bucle publicado. `feature_index.tsv` hace explícito el mapeo.
- **DEC-010 — Arquitectura.** Se reconstruye 1→512→32 por concordancia artículo/checkpoint. Se comparte una instancia StarReLU como en el patrón del fuente. El conteo obligatorio es 141,732; 141,704 se registra como discrepancia.
- **DEC-011 — Dos resultados.** `forensic_test_selected` usa test para selección y se etiqueta sesgado. `valid_subject_disjoint` crea validación estratificada sólo dentro de train y toca test una vez.
- **DEC-012 — Métrica.** Balanced accuracy por sujeto es primaria. Se promedia `p_AD` y se usa umbral fijo 0.5. Las métricas por ventana son secundarias.
- **DEC-013 — Entorno.** No se entrena localmente. Preparación real y entrenamiento real se reservan a Kaggle. Los tests locales, si existen, no son resultados científicos.
- **DEC-014 — Almacenamiento.** Datos, características, checkpoints y predicciones completas quedan fuera de Git. Sólo se versiona el paquete reducido.
- **DEC-015 — Resultados pendientes.** No se insertan números hasta completar tres semillas en Kaggle. Un benchmark de 10 épocas sólo estima costo.
- **DEC-016 — GitHub.** La creación remota queda pendiente de autenticación oficial de GitHub CLI. El correo debe verificarse en GitHub antes del primer push; no se solicitan tokens en texto.
