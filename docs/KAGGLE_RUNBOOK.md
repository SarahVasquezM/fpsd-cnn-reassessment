# Guía de ejecución en Kaggle

Esta guía asume poca experiencia de infraestructura. No requiere VS Code ni una instalación local de Python.

## 1. Abrir los notebooks

Entre a Kaggle, elija **Code → New Notebook → File → Import Notebook** e importe primero `00_prepare_features_kaggle.ipynb`. Repita después con `01_train_baseline_kaggle.ipynb`. En ambos casos añada el repositorio como fuente de código o suba una copia del repositorio a la sesión, de modo que la carpeta que contiene `pyproject.toml` esté disponible.

## 2. Añadir la fuente OpenNeuro

Use CPU; no necesita GPU. Si la cuenta tiene Internet habilitado, el notebook puede descargar los 65 derivados AD/CN de OpenNeuro v1.0.9. Si Internet está deshabilitado, añada el dataset Kaggle `thngdngvn/openneuro-ds004504`; el notebook detecta `ds004504/participants.tsv` y `ds004504/derivatives/` y ejecuta con `--skip-download`. En ambos casos se verifican los SHA-256 de los `.set`.

## 3. Preparar y guardar las características

Pulse **Save Version → Save & Run All**. Al terminar, abra la sección **Output**. Deben existir:

- `/kaggle/working/fpsd_features/`;
- `/kaggle/working/fpsd_features_bundle.zip`.

Seleccione **Create Dataset** desde el output, marque el dataset como **Private** y use el nombre `fpsd-paper95-features`. No cambie archivos dentro del paquete.

## 4. Añadir características y seleccionar GPU

Abra el notebook 01. En **Add Input** agregue `fpsd-paper95-features` y el dataset privado `fpsd-cnn-reassessment-code`. El paquete de código debe contener `SOURCE_COMMIT`; la primera celda rechaza versiones antiguas que no puedan demostrar su procedencia. En **Notebook options → Accelerator** seleccione una GPU. Internet puede permanecer apagado.

## 5. Entrenar

Pulse **Save Version → Save & Run All**. El notebook verifica hashes, procedencia y GPU, ejecuta las pruebas y hace un benchmark de 10 épocas. Si estima que dos protocolos × tres semillas caben con 10% de margen en menos de 10 horas, continúa en esa versión. Si no, la versión ejecuta una sola semilla; cree dos versiones adicionales cambiando únicamente `SEEDS` a `[2027]` y `[2028]`. No cambie ningún otro parámetro. Una ejecución completa debe dejar 12 pruebas aprobadas y `results/metrics/github_commit.txt` con un hash Git de 40 caracteres.

## 6. Encontrar y descargar outputs

Los resultados completos quedan en `/kaggle/working/week1_results/`. Descargue:

- `week1_results_bundle.zip`: checkpoints, predicciones completas, historiales y ambiente;
- `github_results_bundle.zip`: sólo métricas, figuras, reporte y manifiestos pequeños.

## 7. Incorporar el paquete reducido al repositorio

Descargue `github_results_bundle.zip`, descomprímalo en la raíz del repositorio y revise el diff antes de confirmar. El paquete respeta las carpetas `results/`, `reports/` y `logs/`. No copie el bundle completo, checkpoints, `.set`, `.npz` o predicciones masivas a GitHub.

La sincronización automática es opcional y está desactivada. Si se habilita más adelante, use un token fino en **Kaggle Secrets**, nunca en una celda, archivo o URL, y empuje sólo a `kaggle/week1-results`, nunca a `main`.

## 8. Reanudar una interrupción

El notebook 00 reusa archivos ya descargados y verificados. En entrenamiento, cada semilla guarda sus artefactos en `week1_results/<protocolo>/seed_<n>/`. Conserve los outputs de la versión interrumpida como dataset privado y ejecute sólo las semillas incompletas sin alterar la configuración.

## 9. Qué nunca modificar manualmente

No edite el split, hashes, orden de canales, bandas, umbral 0.5, semillas ni resultados. No renombre sujetos o ventanas. No copie credenciales en notebooks. No describa el benchmark o una ejecución parcial como entrenamiento completo.

## 10. VS Code opcional

Si ya está instalado, puede abrir la carpeta del repositorio, revisar archivos y diferencias, inspeccionar notebooks y administrar Git visualmente. No configure Python, CUDA, MNE o PyTorch localmente y no intente conectar VS Code con Kaggle durante la semana 1.

## 11. Semana 2: bundle de 88 sujetos

La guía completa y operativa de semana 2 está en [`docs/WEEK2_RUNBOOK.md`](WEEK2_RUNBOOK.md). Las secciones siguientes son sólo un resumen.

Importe `02_prepare_week2_features_kaggle.ipynb`, adjunte el mismo mirror de OpenNeuro y use CPU. La salida válida es `/kaggle/working/fpsd_features_all_bundle.zip`; debe contener 36 AD, 29 CN y 23 FTD. Guárdela como dataset privado `fpsd-paper95-all-features`.

## 12. Semana 2: benchmark y entrenamiento

Importe `03_train_week2_kaggle.ipynb`, adjunte el paquete de código y `fpsd-paper95-all-features`, y active GPU. Mantenga primero `MODE = 'benchmark'`: se miden cinco folds LOSO de AD–CN y cinco de CN–FTD con 10 épocas. Si una semilla completa no cabe con margen en nueve horas, divida la ejecución con `FOLD_START` inclusivo y `FOLD_STOP` exclusivo.

Después use `MODE = 'run'`. El orden aprobado es:

1. `published_holdout` para CN–FTD, balance `none`, semillas 2026–2028; el modo automático ejecuta reconstrucción forense y evaluación válida.
2. `grouped5` para AD–CN y CN–FTD, balances `none` y `subject_class`, semillas 2026–2028.
3. `loso` para AD–CN y CN–FTD, balance `none`, sólo semilla 2026 inicialmente.

Conserve `--resume`: un fold terminado sólo se reutiliza si coinciden configuración, sujetos y hash del bundle. Un directorio incompleto o una firma distinta detiene la ejecución en vez de sobrescribir resultados.

No compare directamente el 93.94% por sujeto de semana 1 con el 88.36% por ventana del artículo. La tabla comparable se genera en `week2_results/article_comparison.csv`; las métricas agrupadas y LOSO responden una pregunta de generalización distinta.
