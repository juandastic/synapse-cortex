# Pruebas de memoria y generación

Usar Python 3.12. La suite corre sin Neo4j, Google, OpenRouter o Notion. Monta el router FastAPI real sin iniciar el lifespan de producción y sustituye los servicios externos con respuestas controladas. Cualquier conexión de red inesperada falla.

## Preparación y comandos

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
make verify                   # formato + lint + todos los tests
make format                   # aplica formato
make lint-fix                 # aplica fixes seguros; revisar el diff
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -k 'retrieval_failure'
.venv/bin/python -m pytest tests/test_api_contracts.py -q
.venv/bin/python -m pytest --cov=app --cov-report=html --cov-report=xml -q
```

Con uv, `uv venv --python 3.12` y `uv pip install -r requirements-dev.lock` sirven para crear el entorno e instalar. `requirements-dev.txt` declara las dependencias y `requirements-dev.lock` fija la resolución completa para las pruebas locales. Los requisitos de despliegue permanecen en `requirements.txt`. Para actualizar el lock de desarrollo, ejecutar `uv pip compile requirements-dev.txt --python .venv/bin/python --output-file requirements-dev.lock` y validar la suite en un entorno limpio.

pytest recoge las pruebas `unittest` existentes de grounding, cache y OpenRouter, y los nuevos tests con fixtures y async. La configuración está en `pyproject.toml`. La cobertura es opcional y no tiene umbral global. `htmlcov/index.html` permite inspeccionar los huecos. Por ahora los checks se ejecutan manualmente en local: antes de cada commit, correr `make verify`. No hay workflows de CI ni hooks de commit; el despliegue no ejecuta la suite automáticamente.

## Checks para desarrollo con agentes

Ruff reúne lint, orden de imports y formato. Las reglas detectan errores de Python, imports y variables sin uso, patrones propensos a bugs y excepciones encadenadas sin causa explícita. El formato usa Python 3.12 y líneas de 100 caracteres. FastAPI `Depends`, `Header` y `Query` se reconocen como valores válidos en los parámetros por defecto.

`make verify` no modifica archivos y ejecuta los checks sobre `app` y `tests`. Usa `.venv/bin/python`; para otro entorno, ejecutar `make verify PYTHON=/ruta/al/python`. También existen `make lint`, `make format-check` y `make test`. Ruff está fijado en el lock de desarrollo y no se añade a los requisitos de producción.

Antes de implementar, definir el comportamiento y el fallo que debe prevenirse. Para cambios relevantes, añadir pruebas sobre resultados, aislamiento de usuario o recuperación; después correr `make verify`, revisar el diff y ejecutar `git diff --check`. No desactivar reglas para ocultar errores. Reportar las comprobaciones realizadas y distinguir pruebas locales de validación con Neo4j/proveedores reales.

El `AGENTS.md` de esta instalación remite a estas instrucciones y conserva su carácter local. Las pautas compartidas viven aquí para que también sirvan en otros checkouts.

## Comportamientos protegidos

| Archivo | Riesgo cubierto |
| --- | --- |
| `test_api_contracts.py` | Autenticación del API, polling en procesamiento, memoria y cache al completar, limpieza terminal, conservación del job cuando falla hidratación o cache |
| `test_graph_rag.py` | Búsqueda aislada por usuario, exclusión de recuerdos compilados, preservación de persona e historia, clientes legacy, contexto de follow-ups y chat disponible si falla Neo4j |
| `test_hydration_budget.py` | Grafo completo cuando cabe, selección por prioridad cuando no cabe, metadata de recuerdos realmente incluidos, presupuesto de contenido y grafo vacío |
| `test_ingestion.py` | No duplicar procesamiento, descartar sesiones insuficientes, conservar detalles del usuario, limitar ruido del asistente y hacer pollable un fallo de Graphiti |
| Tests previos | Grounding de Gemini, configuración de cache, payloads y errores de OpenRouter |

Las pruebas de hidratación ejecutan el engine completo con un driver que retorna filas controladas. No verifican que Neo4j ejecute correctamente las consultas Cypher. El presupuesto actual cuenta el contenido de recuerdos; los títulos y separadores añaden caracteres al texto final.

## Escribir pruebas útiles

- Usar el router real mediante ASGI cuando el contrato HTTP importe. No arrancar `app.main`, porque inicializa servicios reales.
- Ejecutar la lógica real de ingestión, retrieval o compilación y mockear sus llamadas a Neo4j/proveedores.
- Comprobar hechos conservados, hechos omitidos, estado terminal y aislamiento por `user_id`. No comparar respuestas completas de un LLM contra una cadena exacta.
- Dar a los jobs IDs nuevos por test y limpiarlos con las funciones públicas del store.
- Para un bug, reproducir primero el fallo con un test. Usar nombres que expliquen qué debe seguir funcionando.

## Pendientes por riesgo

El algoritmo de hidratación todavía necesita casos sobre rollover entre todas las categorías, relaciones recientes/estables y agrupación de episodios por fecha. La suite tampoco cubre todavía exportaciones y correcciones Notion ni recovery ante reinicio del store en memoria. Los contratos reales de Neo4j, caches Gemini y APIs de proveedores requieren otra capa de integración. La calidad de extracción y recall necesita evaluaciones con datos representativos, separadas de estos tests deterministas.

La estrategia del frontend y Convex está en `synapse-chat-ai/TESTING.md`, en el repo hermano.

## Referencias

- [pytest: ejecución de suites unittest existentes](https://docs.pytest.org/en/stable/how-to/unittest.html)
- [FastAPI: pruebas async con ASGI](https://fastapi.tiangolo.com/advanced/async-tests/)
