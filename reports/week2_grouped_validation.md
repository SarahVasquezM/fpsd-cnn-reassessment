# Semana 2 · validación agrupada y LOSO verdadero

## Alcance

La implementación usa exclusivamente los EEG preprocesados de OpenNeuro `ds004504` v`1.0.9`. Mantiene la arquitectura FPSD-CNN, las ventanas de 5 s, los 95 rasgos FPSD, RMSprop, `learning_rate=0.0005`, `batch_size=64`, 100 épocas y umbral 0.5. No incorpora EEG crudo, demografía ni MMSE.

## Punto de comparación congelado

Los porcentajes comparables con el artículo son exactitud **por ventana** sobre la partición fija publicada:

| Tarea | Artículo, preprocesado | Reproducción semana 1 |
|---|---:|---:|
| AD–CN | 88.36% | 85.50% ± 1.71% con selección forense; 83.52% ± 0.21% con validación limpia |
| CN–FTD | 82.22% | 75.95% ± 1.12% con selección forense; 69.15% ± 1.70% con validación limpia |

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

## Resultados ejecutados con la semilla 2026

Todas las cifras de esta tabla son **por sujeto**. Cada sujeto aparece exactamente una vez en las predicciones fuera de muestra que se agregan para calcular las métricas.

| Tarea | Evaluación | Balance | Accuracy | Balanced accuracy | ROC AUC | IC95% bootstrap de accuracy |
|---|---|---|---:|---:|---:|---:|
| AD–CN | LOSO | ninguno | 76.92% | 77.16% | 84.29% | 66.15–86.15% |
| AD–CN | grouped5 | ninguno | 80.00% | 80.27% | 86.69% | 69.23–89.23% |
| AD–CN | grouped5 | `subject_class` | 80.00% | 80.60% | 87.16% | 69.23–89.23% |
| CN–FTD | LOSO | ninguno | 76.92% | 76.61% | 81.11% | 65.38–88.46% |
| CN–FTD | grouped5 | ninguno | 71.15% | 70.54% | 78.71% | 57.69–82.69% |
| CN–FTD | grouped5 | `subject_class` | 71.15% | 71.44% | 75.71% | 57.69–82.69% |

Los intervalos se calcularon con 10,000 remuestreos de las predicciones ya obtenidas por sujeto; el bootstrap no vuelve a entrenar ni a ejecutar el modelo.

## Holdout publicado CN–FTD · tres semillas

Estas cifras son accuracy **por ventana** sobre la partición fija del artículo: 35 sujetos de entrenamiento (20 CN y 15 FTD) y 17 de prueba (9 CN y 8 FTD).

| Protocolo | 2026 | 2027 | 2028 | Media ± DE | Diferencia frente al artículo |
|---|---:|---:|---:|---:|---:|
| Forense | 76.26% | 76.88% | 74.70% | 75.95% ± 1.12 pp | −6.27 pp |
| Válido | 67.21% | 70.37% | 69.86% | 69.15% ± 1.70 pp | −13.07 pp |

La variante forense es la reconstrucción más cercana al mecanismo del código publicado, pero selecciona el checkpoint observando el test y por eso está sesgada. El protocolo válido elige el checkpoint con una validación interna formada sólo a partir del entrenamiento y consulta el test una vez; es metodológicamente más limpio, aunque ya no reproduce exactamente el procedimiento del artículo.

Como referencia secundaria, la accuracy media por sujeto fue 80.39% en la variante forense y 74.51% en la válida. Estas cifras no se comparan directamente con el 82.22% del artículo porque la unidad es distinta.

### Lectura para la presentación

- En AD–CN, grouped5 sin balanceo superó a LOSO en 3.11 puntos de balanced accuracy. El balanceo añadió sólo 0.34 puntos y dejó el accuracy en 80.00%.
- En CN–FTD, grouped5 sin balanceo quedó 6.07 puntos por debajo de LOSO en balanced accuracy. El balanceo recuperó 0.90 puntos, pero dejó el accuracy en 71.15% y redujo el ROC AUC.
- El balanceo no produjo una mejora grande ni uniforme. Con una sola semilla, el resultado responsable es describirlo como una señal pequeña e inconclusa, no como una mejora demostrada.
- grouped5 y LOSO son evaluaciones separadas. En grouped5 se entrenan cinco modelos y cada uno prueba aproximadamente una quinta parte de los sujetos. En LOSO se entrena un modelo por sujeto y cada modelo prueba exactamente a una persona. No se ejecutó LOSO dentro de cada fold.
- Los intervalos son amplios y se solapan. Por eso no hay evidencia suficiente para declarar que grouped5 sea intrínsecamente mejor que LOSO; la diferencia observada depende de la tarea y de cómo cambia el conjunto de entrenamiento en cada partición.

## Ejecución auditada

1. Características para los 88 sujetos: completadas.
2. LOSO AD–CN sin balanceo, semilla 2026: 65/65 folds completos.
3. LOSO CN–FTD sin balanceo, semilla 2026: 52/52 folds completos.
4. grouped5, semilla 2026: 20/20 entrenamientos completos — ambas tareas, con y sin balanceo.
5. Holdout publicado CN–FTD: 6/6 entrenamientos completos — protocolos forense y válido, semillas 2026, 2027 y 2028.
6. Pendientes confirmatorios: semillas 2027 y 2028 para grouped5.
7. LOSO balanceado permanece condicionado a una mejora consistente en grouped5; los resultados actuales no satisfacen claramente ese criterio.

## Artefactos esperados

Cada fold conserva historial, predicciones por ventana y sujeto, métricas, manifiesto de sujetos, tiempo, checkpoint y firma de reanudación. Los agregados incluyen predicciones OOF, balanced accuracy, accuracy, sensibilidad, especificidad, macro-F1, ROC-AUC, PR-AUC, MCC e intervalos bootstrap de 10,000 remuestreos por sujeto.

## Estado

La matriz grouped5 se ejecutó correctamente en el notebook privado `sarahvasquez97/03-train-week2-kaggle`, versión 5, ejecución `362686497`, con GPU Tesla T4. Kaggle informó estado exitoso y 36 minutos de ejecución total; los 20 entrenamientos acumularon 1,650.52 s. Esos resultados siguen siendo observaciones de una sola semilla y no deben presentarse como promedio multisemilla.

El holdout CN–FTD se ejecutó en el mismo notebook, versión 6, ejecución `362696458`, también con Tesla T4. Kaggle informó estado exitoso en 1,064.3 s; pasaron 21 pruebas y se completaron los seis entrenamientos previstos. La trazabilidad de la descarga, hashes y métricas está en `results/week2/provenance/cn_ftd_published_holdout_seeds2026_2028.json`.
