# API

Run:

`uvicorn src.api.main:app --reload`

Swagger: `/docs`

## POST /ingest/live
Discovers live news and saves the runtime cache.

## POST /pipeline/run-live
Discovers live news and immediately runs the complete risk pipeline.

Body:

```json
{"topic":"India ports and supply chain disruptions","provider":"both","hours":48,"limit":8}
```

## GET /news/live
Returns the last live retrieval.

## GET /events
Returns events from the last live run.

## GET /alerts
Returns generated alerts.

## POST /feedback
Stores analyst feedback.
