# Guía de ejecución en Kaggle

Esta guía asume poca experiencia de infraestructura. No requiere VS Code ni una instalación local de Python.

## 1. Abrir los notebooks

Entre a Kaggle, elija **Code → New Notebook → File → Import Notebook** e importe primero `00_prepare_features_kaggle.ipynb`. Repita después con `01_train_baseline_kaggle.ipynb`. En ambos casos añada el repositorio como fuente de código o suba una copia del repositorio a la sesión, de modo que la carpeta que contiene `pyproject.toml` esté disponible.

## 2. Activar Internet para la preparación

En el notebook 00 abra **Notebook options** y active **Internet on**. Este paso descarga dependencias y sólo los 65 archivos derivados AD/CN de OpenNeuro v1.0.9. Use CPU; no necesita GPU.

## 3. Preparar y guardar las características

Pulse **Save Version → Save & Run All**. Al terminar, abra la sección **Output**. Deben existir:

- `/kaggle/working/fpsd_features/`;
- `/kaggle/working/fpsd_features_bundle.zip`.

Seleccione **Create Dataset** desde el output, marque el dataset como **Private** y use el nombre `fpsd-paper95-features`. No cambie archivos dentro del paquete.

## 4. Añadir características y seleccionar GPU

Abra el notebook 01. En **Add Input** busque su dataset privado `fpsd-paper95-features` y agréguelo. En **Notebook options → Accelerator** seleccione una GPU. Internet puede permanecer apagado si el repositorio y dependencias ya están disponibles; si la primera celda informa paquetes faltantes, actívelo sólo para instalar y vuelva a ejecutar.

## 5. Entrenar

Pulse **Save Version → Save & Run All**. El notebook verifica hashes y GPU, ejecuta pruebas rápidas y hace un benchmark de 10 épocas. Si estima que dos protocolos × tres semillas caben con 10% de margen en menos de 10 horas, continúa en esa versión. Si no, la versión ejecuta una sola semilla; cree dos versiones adicionales cambiando únicamente `SEEDS` a `[2027]` y `[2028]`. No cambie ningún otro parámetro.

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
