# Runbook de semana 2 — FPSD-CNN

Esta guía permite reproducir la semana 2 desde Kaggle sin instalar Python, CUDA ni MNE en Windows. El flujo usa exclusivamente los EEG **preprocesados** de OpenNeuro `ds004504` v`1.0.9`.

## 1. Qué se implementó

- Un bundle único con 88 sujetos: 36 AD, 29 CN y 23 FTD.
- La partición fija CN–FTD publicada: 20 CN + 15 FTD para entrenamiento y 9 CN + 8 FTD para prueba.
- Validación estratificada de cinco folds con separación por sujeto.
- LOSO verdadero: 65 folds AD–CN y 52 folds CN–FTD.
- Balanceo opcional `subject_class`, aplicado únicamente al entrenamiento.
- Ejecuciones por rangos de folds con reanudación y verificación de firmas.
- Predicciones por ventana y sujeto, métricas OOF, MCC, ROC-AUC, PR-AUC e intervalos bootstrap de 10,000 remuestreos por sujeto.
- Paquetes de resultados completo y reducido para GitHub.

El modelo, los 95 rasgos FPSD, las ventanas de 5 s, RMSprop, `learning_rate=0.0005`, `batch_size=64`, 100 épocas y el umbral 0.5 permanecen sin cambios.

## 2. Archivos que se utilizarán

1. `notebooks/02_prepare_week2_features_kaggle.ipynb`: prepara los 88 sujetos con CPU.
2. `notebooks/03_train_week2_kaggle.ipynb`: hace el benchmark y los entrenamientos con GPU.
3. `fpsd-cnn-reassessment-code-v8.zip`: paquete de código para actualizar el dataset privado de Kaggle.

El ZIP de código coloca `pyproject.toml` en la raíz. Esto evita el error anterior en el que el notebook extraía una carpeta adicional y después no encontraba el repositorio.

## 3. Actualizar el dataset privado de código

En Kaggle abra el dataset privado `sarahvasquez97/fpsd-cnn-reassessment-code` y cree una versión nueva con `fpsd-cnn-reassessment-code-v8.zip`.

No descomprima ni reorganice manualmente el ZIP. El paquete contiene:

- `SOURCE_TREE_SHA256.txt`, con la identidad del código científico;
- `CODE_SHA256SUMS.txt`, con el hash de cada archivo incluido.

El entrenamiento se detiene si el código extraído no coincide con la identidad declarada.

## 4. Preparar las características de los 88 sujetos

1. En Kaggle seleccione **Code → New Notebook → File → Import Notebook**.
2. Importe `02_prepare_week2_features_kaggle.ipynb`.
3. En **Add Input** agregue:
   - `thngdngvn/openneuro-ds004504`;
   - el dataset privado `fpsd-cnn-reassessment-code` actualizado.
4. En **Notebook options** deje `Accelerator = None`. Internet puede permanecer apagado.
5. Ejecute **Save Version → Save & Run All**.

Esta etapa utiliza CPU y puede tardar varias horas porque calcula Welch/FPSD para los 88 sujetos. Un uso alto de CPU es esperado.

### Comprobaciones obligatorias

La última celda debe mostrar exactamente:

```text
{'AD': 36, 'CN': 29, 'FTD': 23}
```

También deben existir:

```text
/kaggle/working/fpsd_features_all/
/kaggle/working/fpsd_features_all_bundle.zip
```

Desde **Output → Create Dataset**, guarde el resultado como dataset **Private** con el nombre:

```text
fpsd-paper95-all-features
```

No modifique archivos dentro del bundle.

También es válido conectar directamente el output exitoso del notebook 02 como **Notebook Output** del notebook 03. Esta fue la ruta usada en la ejecución auditada del 23 de octubre de 2026; evita duplicar el bundle y conserva la versión exacta de código que produjo las características.

## 5. Configurar el notebook de entrenamiento

1. Importe `03_train_week2_kaggle.ipynb`.
2. En **Add Input** agregue:
   - `fpsd-cnn-reassessment-code`;
   - `fpsd-paper95-all-features`.
3. En **Notebook options → Accelerator**, seleccione una GPU.
4. Internet puede permanecer apagado.
5. Ejecute primero el notebook sin cambiar su configuración inicial.

La primera ejecución usa:

```python
MODE = 'benchmark'
```

El notebook valida los hashes, confirma 36 AD + 29 CN + 23 FTD, ejecuta las pruebas y mide 5 folds LOSO de 10 épocas para cada tarea.

Los resultados del benchmark quedan en:

```text
benchmark_ad_cn_loso_none.json
benchmark_cn_ftd_loso_none.json
```

Use `estimated_hours_full_seed` para decidir el tamaño de cada versión. Si `fits_9h_with_margin` es `false`, divida LOSO en rangos más pequeños.

### Ejecuciones auditadas del 23 de octubre de 2026

- Preparación de características: notebook `02_prepare_week2_features_kaggle_fixed`, versión 2, ejecución `361545939`, exitosa en 334.9 s; produjo 66 archivos para 88 sujetos.
- Benchmark GPU: notebook `03_train_week2_kaggle`, versión 2, ejecución `361550229`, exitosa en 164.4 s con Tesla T4.
- Estimación LOSO AD–CN, semilla completa: 2.0904 h; `fits_9h_with_margin=true`.
- Estimación LOSO CN–FTD, semilla completa: 1.2808 h; `fits_9h_with_margin=true`.
- LOSO AD–CN semilla 2026, folds `[0, 65)`: versión 3, ejecución `361551285`, completó los 65 entrenamientos en 7,189.35 s de GPU. Kaggle marcó la versión como error porque la agregación final intentó ordenar el objeto estructurado `fold`; no se perdió ningún fold.
- Los artefactos de los 65 folds se descargaron y la agregación se reconstruyó con 65 filas. Resultado LOSO por sujeto: accuracy 76.92%, balanced accuracy 77.16% y ROC AUC 84.29% sobre 65 sujetos. El detalle auditable está en `results/week2/`.
- La corrección de agregación está en el commit `7a79e8c` y en `fpsd-cnn-reassessment-code-v8.zip`. La prueba de regresión acepta tanto manifiestos estructurados como el formato plano anterior.
- LOSO CN–FTD semilla 2026, folds `[0, 52)`: versión 4 lanzada con el paquete v8 y GPU T4; consulte su estado antes de declarar resultados.

La ejecución automatizada actual comienza por LOSO con una sola semilla, tal como se aprobó. Las etapas de holdout y grouped5 siguen formando parte de la matriz completa, pero sus resultados no deben declararse terminados hasta que existan versiones exitosas y artefactos descargados.

## 6. Orden exacto de ejecución

Después del benchmark cambie:

```python
MODE = 'run'
```

### Etapa A — holdout publicado CN–FTD

```python
TASK = 'cn_ftd'
EVALUATION = 'published_holdout'
BALANCE = 'none'
SEEDS = [2026, 2027, 2028]
FOLD_START = 0
FOLD_STOP = 1
```

El protocolo automático ejecuta dos variantes:

- `forensic`: selecciona el checkpoint observando el test, como el código publicado; es sesgado, pero permite la comparación forense.
- `valid`: selecciona el checkpoint con validación interna y evalúa el test una sola vez.

### Etapa B — validación agrupada de cinco folds

Ejecute las cuatro combinaciones siguientes. En todas use `SEEDS = [2026, 2027, 2028]`, `FOLD_START = 0` y `FOLD_STOP = 5`.

| Ejecución | `TASK` | `EVALUATION` | `BALANCE` |
|---|---|---|---|
| 1 | `ad_cn` | `grouped5` | `none` |
| 2 | `ad_cn` | `grouped5` | `subject_class` |
| 3 | `cn_ftd` | `grouped5` | `none` |
| 4 | `cn_ftd` | `grouped5` | `subject_class` |

### Etapa C — LOSO verdadero, primero una semilla

Empiece siempre con:

```python
EVALUATION = 'loso'
BALANCE = 'none'
SEEDS = [2026]
```

Para AD–CN deben completarse los folds `[0, 65)`. Para CN–FTD deben completarse `[0, 52)`.

Si el benchmark no permite ejecutar todos en una versión, divida sin solapamientos. Ejemplo:

```text
AD–CN:  [0, 15), [15, 30), [30, 45), [45, 65)
CN–FTD: [0, 15), [15, 30), [30, 45), [45, 52)
```

Los rangos son ejemplos; ajústelos con el benchmark real. `FOLD_START` es inclusivo y `FOLD_STOP` exclusivo.

No ejecute LOSO con `subject_class` hasta comprobar que el balanceo mejora de forma consistente los cinco folds en ambas tareas.

## 7. Cómo reanudar sin perder resultados

El notebook añade `--resume`. Antes de empezar una nueva versión:

1. Guarde el output de la versión anterior como dataset privado.
2. Añádalo como input o copie su contenido a `/kaggle/working/week2_results` antes de ejecutar el siguiente rango.
3. Mantenga iguales tarea, evaluación, balance, semilla y configuración.

Cada fold tiene una firma que incluye sujetos, configuración, hash de características e identidad del código. Un fold completo se omite. Un fold incompleto o con firma distinta detiene la ejecución; no se sobrescribe silenciosamente.

## 8. Resultados esperados

La raíz `/kaggle/working/week2_results/` contiene:

```text
article_comparison.csv
environment.json
source_tree_sha256.txt
manifests/
ad_cn/
cn_ftd/
SHA256SUMS.txt
```

Dentro de cada tarea/evaluación/balance se generan:

```text
metrics_by_fold.csv
metrics_by_seed.csv
aggregate/<protocol>/seed_<n>/oof_window_predictions.parquet
aggregate/<protocol>/seed_<n>/oof_subject_predictions.csv
aggregate/<protocol>/seed_<n>/subject_bootstrap_95.json
```

Cada fold conserva `history.csv`, métricas, predicciones, manifiesto, tiempo, checkpoint y `completed.json`.

La última celda crea:

- `week2_results_bundle.zip`: artefactos completos, incluidos checkpoints y predicciones;
- `week2_github_results_bundle.zip`: paquete reducido sin checkpoints, Parquet ni historiales.

## 9. Qué número comparar con el artículo

El artículo reporta exactitud **por ventana**:

- AD–CN preprocesado: 88.36%;
- CN–FTD preprocesado: 82.22%.

El archivo `article_comparison.csv` usa esa misma unidad. El 93.94% de semana 1 es una métrica **por sujeto** y no debe presentarse como equivalente al 88.36% del artículo.

Las métricas agrupadas y LOSO evalúan generalización entre sujetos y responden una pregunta distinta; deben presentarse por separado.

## 10. Cuándo una ejecución está completa

No declare resultados finales hasta comprobar:

- holdout CN–FTD completo en tres semillas;
- cinco folds completos para cada tarea, balance y semilla;
- 65 sujetos OOF en LOSO AD–CN;
- 52 sujetos OOF en LOSO CN–FTD;
- ausencia de errores de hash o firma;
- `SHA256SUMS.txt` presente;
- métricas agregadas calculadas sobre todos los sujetos esperados.

## 11. Errores frecuentes

### No se encuentra `pyproject.toml`

Actualice el dataset privado con `fpsd-cnn-reassessment-code-v8.zip`. El notebook selecciona de forma natural la carpeta o ZIP con el número de versión más alto y el paquete tiene el repositorio en la raíz.

### Se encuentran varios bundles de características

Quite inputs antiguos del notebook 03 y deje únicamente `fpsd-paper95-all-features`.

### La GPU no está disponible

Active un acelerador en **Notebook options** y reinicie la sesión.

### La sesión no alcanza a terminar LOSO

Use rangos de folds más pequeños y conserve el output como dataset privado entre versiones.

### Error de firma al reanudar

No mezcle resultados producidos con distintos bundles, código, semillas o configuraciones. Abra una carpeta de resultados nueva para la configuración diferente.

## 12. Evidencia de implementación

Antes de empaquetar la implementación inicial se ejecutaron 18 pruebas correctamente; una prueba dependiente de GPU quedó omitida localmente de forma esperada. Después de detectar el fallo de agregación se añadieron dos pruebas de regresión y se verificaron junto con las pruebas de reproducibilidad (4 correctas). El entrenamiento científico se ejecuta exclusivamente en Kaggle.
