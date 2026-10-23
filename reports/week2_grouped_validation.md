# Semana 2 · validación agrupada y LOSO verdadero

## Alcance

La implementación usa exclusivamente los EEG preprocesados de OpenNeuro `ds004504` v`1.0.9`. Mantiene la arquitectura FPSD-CNN, las ventanas de 5 s, los 95 rasgos FPSD, RMSprop, `learning_rate=0.0005`, `batch_size=64`, 100 épocas y umbral 0.5. No incorpora EEG crudo, demografía ni MMSE.

## Punto de comparación congelado

Los porcentajes comparables con el artículo son exactitud **por ventana** sobre la partición fija publicada:

| Tarea | Artículo, preprocesado | Reproducción semana 1 |
|---|---:|---:|
| AD–CN | 88.36% | 85.50% ± 1.71% con selección forense; 83.52% ± 0.21% con validación limpia |
| CN–FTD | 82.22% | pendiente de ejecución en Kaggle |

El `93.94%` de semana 1 es una métrica **por sujeto** y no se presenta como equivalente directo al porcentaje por ventana del artículo.

## Cohortes y particiones

- Bundle completo: 88 sujetos, 36 AD + 29 CN + 23 FTD.
- Tarea AD–CN: 65 sujetos.
- Tarea CN–FTD: 52 sujetos.
- Holdout CN–FTD publicado: entrenamiento 20 CN + 15 FTD; prueba 9 CN + 8 FTD.
- Validación agrupada: cinco folds estratificados y deterministas, con separación por sujeto.
- LOSO verdadero: 65 folds AD–CN y 52 folds CN–FTD; cada sujeto aparece exactamente una vez en prueba.

La validación interna se obtiene únicamente de los sujetos del entrenamiento exterior. La selección forense que inspecciona el test se conserva sólo para reconstruir el holdout publicado y siempre queda marcada como sesgada.

## Balanceo

`subject_class` asigna a cada ventana un peso inverso al número de ventanas de su sujeto y al número de sujetos de su clase. Así, cada clase tiene la misma masa total de muestreo y todos los sujetos de una clase tienen la misma masa. La función de pérdida continúa siendo entropía cruzada sin ponderar. Validación y prueba nunca se remuestrean.

## Orden de ejecución

1. Generar `fpsd_features_all_bundle.zip` con `02_prepare_week2_features_kaggle.ipynb` y publicarlo como dataset privado `fpsd-paper95-all-features`.
2. Ejecutar el benchmark de cinco folds LOSO para ambas tareas desde `03_train_week2_kaggle.ipynb`.
3. Ejecutar el holdout CN–FTD, primero forense y después válido, con semillas 2026, 2027 y 2028.
4. Ejecutar cinco folds para ambas tareas, con `none` y `subject_class`, en las tres semillas.
5. Ejecutar LOSO sin balanceo con una sola semilla, 2026. Dividirlo en rangos de folds si el benchmark estima más de nueve horas.
6. Ejecutar LOSO balanceado sólo si los cinco folds muestran una mejora consistente.

## Artefactos esperados

Cada fold conserva historial, predicciones por ventana y sujeto, métricas, manifiesto de sujetos, tiempo, checkpoint y firma de reanudación. Los agregados incluyen predicciones OOF, balanced accuracy, accuracy, sensibilidad, especificidad, macro-F1, ROC-AUC, PR-AUC, MCC e intervalos bootstrap de 10,000 remuestreos por sujeto.

## Estado

La implementación superó 18 pruebas locales; una prueba dependiente de GPU quedó omitida de forma esperada. Los resultados científicos de semana 2 permanecen pendientes hasta ejecutar los notebooks en Kaggle GPU; no se imputan ni anticipan cifras.
