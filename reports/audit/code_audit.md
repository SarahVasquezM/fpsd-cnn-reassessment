# Auditoría del código FPSD-CNN publicado

Fecha de verificación: 2026-09-26
Upstream: `changjingzhi/FPSD-CNN`
Commit: `45f33193c8b100ec715b8606aadaf42c08027bb8`

## Conclusión

El repositorio publicado no forma un pipeline ejecutable coherente con el artículo: el extractor genera 114 valores, la etapa siguiente acepta sólo 95, el modelo fuente usa 128 filtros iniciales mientras el checkpoint y la descripción requieren 512, y el conjunto llamado test selecciona el mejor checkpoint durante las 100 épocas. Por ello se implementan dos resultados inequívocos: una reconstrucción forense sesgada y un baseline válido con separación por sujeto.

## Evidencia directa

| Verificación | Evidencia observada | Consecuencia |
|---|---|---|
| Split | Una única lista fija: train 25 AD + 18 CN; test 11 AD + 11 CN. | Es hold-out fijo 70/30 por sujeto, no LOSO. |
| Fuga | `test_loader` se pasa como `val_loader` y decide `best_model.pth` cada época. | Selección sobre test; resultado optimista. |
| Extractor | Seis bandas: delta, theta, alpha, low-beta, high-beta, gamma. | 19×6 = 114 valores. |
| Consumidor | Descarta matrices si `data.shape[1] != 95`. | El pipeline literal no llega al entrenamiento. |
| Arquitectura | Fuente principal: Conv1D 1→128→32. | No coincide con artículo/checkpoint 512→32. |
| Parámetros | Reconstrucción 512→32 con StarReLU compartida: 141,732. | El artículo declara 141,704; diferencia de 28. |
| Ventanas | Eventos cada 5 s, épocas `tmin=0`, `tmax=4.0`. | No son ventanas contiguas exactas de 5 s. |
| Derivados | Remuestrea 500→512, filtra 1–45 y referencia promedio. | Reprocesa derivados que ya fueron limpiados. |
| CSV | Escribe sin encabezados; `pandas.read_csv` usa encabezado implícito. | Se pierde la primera ventana de cada CSV. |
| Batches | `drop_last=True` en train y test. | Descarta ventanas de evaluación. |
| Semillas | No fija Python, NumPy, PyTorch, CUDA ni DataLoader. | Corridas no deterministas. |
| Orden | `os.listdir` sin ordenar y clases enumeradas desde directorios. | Orden y etiquetas dependen del sistema de archivos. |
| GPU | Entrenamiento no mueve modelo ni lotes a CUDA. | El script publicado entrena en CPU aunque haya GPU. |

## Hashes SHA-256 de evidencia

- `model/AD-CN.pth`: `b37a8197b8e59f077adbb0a52688d4438f1d97de332a626cc2ae09ffafd16817`
- `1-data_processing/1-psd.py`: `110de88e8dc1bfc63d9e0bab87737774ffe9bd87668a29b5846c2821a0e43399`
- `1-data_processing/2-fpsd-processing.py`: `977edbf5e8544a3e2ed46e8990652a1b356fdcae7725c84985b40e99e5866356`
- `1-data_processing/3-Create-list.py`: `225a69420ff033d4c01e28b64aadf9e128ee3a31167e3f575b244b4db4b47ede`
- `2-model/fpsd-cnn.py`: `56530212c934baae660d16e7174927ee8d7c27857fdb90ada8cf83e5881b78fa`
- `participants.tsv` v1.0.9: `bd17cf02ec9733c9029d4a6d977b2918309b3c667ff9ceb1b5dce6e3cebccbc1`

## Decisión de reconstrucción

`paper95` usa derivados sin nuevo filtrado, referencia ni ICA; conserva 500 Hz; crea ventanas `[inicio, inicio+5 s)`; usa Welch Hann con `n_fft=n_per_seg=256`, `n_overlap=10`, detrend constante; normaliza cada PSD por la suma de bins 0.5–45 Hz; y calcula cinco medias de banda. La vista plana es banda-mayor porque ése es el orden del bucle upstream.

La métrica primaria es balanced accuracy por sujeto después de promediar `p_AD`. Las métricas por ventana se conservan como secundarias y no se interpretan como observaciones independientes.
