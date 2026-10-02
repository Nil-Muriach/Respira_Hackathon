# Declaración de uso de inteligencia artificial

## Herramienta
Claude (Anthropic), a través de Claude Code, el asistente de programación en terminal.

## En qué ayudó
- **Código:** funciones de `src/`, tests con datos sintéticos y celdas de los notebooks.
- **Ejecución de los análisis** en el entorno local del equipo: limpieza, clustering, trayectorias y modelos.
- **Revisión crítica de resultados:** detectó, por ejemplo, que el fenotipo a 3 meses equivale casi a un corte de FVC, que los medoides tienen pruebas ausentes y que la DLCO a 3 meses sola supera a los modelos complejos.
- **Redacción de la documentación** (`README.md`, `docs/`) y de los textos de los notebooks.

## Qué decidió el equipo
- La estrategia: qué cohortes usar, horizontes, dianas y qué fenotipo usar como principal.
- La elección de los modelos finales y qué se entrega.
- La validación clínica de las reglas de limpieza (`docs/registro_decisiones.md`, columna "Validación") y los nombres de los fenotipos.

## Cómo evitamos conclusiones a medida
- Los criterios de validez del pasaporte (prueba nula, estabilidad, replicación) están fijados en `config.yaml` y se aplican igual a todos los fenotipos. Los que no los cumplen se presentan como no replicados.
- En el modelo de la primera visita (notebook 08), la regla de decisión se fijó antes de ajustar los modelos: solo se adopta un modelo más complejo si mejora a la DLCO sola con un IC que excluya el 0.
- **Excepción declarada:** el criterio operativo del sello de trayectorias (notebook 05) se escribió tras una exploración inicial del modelo, así que no es un criterio ciego.
- Los resúmenes de cada notebook se redactaron después de ejecutarlo, con las cifras reales.

## Datos y privacidad
El código se ejecutó en local sobre los datos seudonimizados, que no se subieron a ningún servicio ni al repositorio. Por diseño, los notebooks solo muestran resultados agregados y suprimen las celdas con N < 10 (`src/privacidad.py`). No se visualizaron filas de pacientes individuales.
