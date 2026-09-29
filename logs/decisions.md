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
- **DEC-016 — GitHub.** GitHub CLI se autenticó mediante el flujo oficial. `sarah.vasquezmo@udlap.mx` fue confirmado como correo verificado antes del primer commit. El repositorio remoto quedó privado y no se solicitaron ni almacenaron tokens en texto.
- **DEC-017 — Kaggle pendiente.** El 2026-09-26 la sesión del navegador no estaba autenticada en Kaggle. No se ejecutaron datos reales, benchmark ni entrenamiento y no se generaron métricas; la pantalla oficial de inicio de sesión quedó abierta para la usuaria.

## 2026-09-27

- **DEC-018 — Entrada Kaggle sin Internet.** La cuenta autenticada no tenía habilitado Internet para notebooks sin verificación telefónica. En lugar de mezclar datos o ejecutar descargas locales, se adjuntó el dataset Kaggle `thngdngvn/openneuro-ds004504` (3 GB), cuya raíz contiene `ds004504/participants.tsv` y `ds004504/derivatives/`. `00_prepare_features_kaggle.ipynb` usa esa entrada de solo lectura con `--skip-download`; el manifiesto y los SHA-256 de cada `.set` siguen siendo obligatorios.
- **DEC-019 — Entorno Kaggle observado.** La imagen CPU fijada por Kaggle (`2026-06-30`) ya contiene la pila científica. Para evitar una instalación de red imposible, los notebooks añaden `src/` a `PYTHONPATH` y serializan las versiones efectivamente importadas. No se presenta como equivalencia a los pines solicitados: la divergencia queda registrada en el artefacto de entorno.

## 2026-09-28

- **DEC-020 — Características completas.** El notebook 00 versión 3 terminó en Kaggle: 65/65 sujetos aprobaron QC, sin NaN ni infinitos, y produjo 10,615 ventanas con forma `N×19×5`. Se conservan `subjects_qc.csv` y el manifiesto pequeño, no el tensor pesado.
- **DEC-021 — Tres semillas en GPU.** El benchmark confirmó que dos protocolos por tres semillas cabían con margen. Se ejecutaron 2026, 2027 y 2028 en Tesla T4; 12 pruebas pasaron antes de entrenar y las seis corridas terminaron.
- **DEC-022 — Procedencia empaquetada.** La primera ejecución completa registró `UNCOMMITTED` porque Kaggle no conserva `.git`. Se añadió el fallback `SOURCE_COMMIT`, una prueba de regresión y el paquete de código v5. La ejecución científica final registra `8f8ab4e1f10be70981ce4b9f02ea0539bd4ac1e8` en todas las predicciones y manifiestos.
- **DEC-023 — Versión científica final.** Se toma como ejecución final el notebook Kaggle versión 7 (`scriptVersionId=353772449`), completado en 10 min 50 s. La versión rápida posterior sólo conserva la visualización y no sustituye la corrida batch.
- **DEC-024 — Integridad del paquete.** `github_results_bundle.zip` tiene SHA-256 `78c50d697f3f0e04f5db1f1d7957157fe16d9e76e058282788c18cc9a437d716`. Se verificaron correctamente los 21 archivos de su manifiesto antes de integrarlos al repositorio.
- **DEC-025 — Comparación prudente.** El 93.94% por sujeto se compara descriptivamente con 88.21–88.36% del artículo, pero no se afirma superioridad: cambian la unidad de evaluación, la selección de checkpoint y detalles del pipeline, y el test sólo contiene 22 sujetos.
