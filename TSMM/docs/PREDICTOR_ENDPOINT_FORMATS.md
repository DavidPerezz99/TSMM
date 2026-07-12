# Predictor Endpoint Formats

This document is aligned to the actual eDep service host in `C:\Users\artur\OneDrive\Documentos\eDep-main`.

After reviewing that app, the important serving constraints are:

1. eDep stores one uploaded model file, one optional `requirements.txt`, one optional `ingestion_pipeline.py`, and one `input_schema.json` per service.
2. eDep generates a FastAPI wrapper from `input_schema.json`.
3. For non-image services, that wrapper exposes `POST /{service_name}/predict` and expects a JSON array of objects, not a raw top-level tensor.
4. `process_input(...)` in `ingestion_pipeline.py` receives a Python `list[dict]`.
5. TSMM predictors still need sequence windows internally, but the eDep request body must wrap each window inside an object field.
6. eDep currently uploads only one model file, while TSMM saves model and scaler artifacts separately by default.

That means the previous raw-array contract was not correct for eDep.

## What eDep Actually Expects

From the eDep backend:

1. The schema generator reads `input_schema.json["properties"]` and builds a Pydantic model.
2. The generated route is usually:

```python
@app.post("/{route}/predict")
async def predict(data: List[InputData], token: dict = Depends(verify_token)):
```

3. So the request body for TSMM predictors should be:

```json
[
  {
    "window": [[...], [...], [...]]
  },
  {
    "window": [[...], [...], [...]]
  }
]
```

or, for flattened input:

```json
[
  {
    "vector": [...]
  }
]
```

Each batch item is an object. The sequence data lives inside one property of that object.

The generated eDep service also uses:

1. bearer authentication on the deployed service route
2. response wrapping as `{"prediction": <process_input output>}`

## Critical Packaging Constraint

TSMM currently saves:

1. model file
2. artifacts file with `scaler_X` and `scaler_y`

eDep currently uploads only one model file per service.

So for TSMM predictors to work cleanly in eDep, you should upload a single bundled file that contains:

1. the trained model
2. `scaler_X`
3. `scaler_y`
4. the serving spec

This repo now includes:

1. `scripts/export_edep_bundle.py` to build that single bundle
2. `scripts/edep_ingestion_pipeline.py` as the eDep-ready pipeline template

Use a `.pkl` output name for the bundle because the eDep UI file picker already allows `.pkl`.

## Available `top1` Predictors

Found in the repo:

1. [config/high10mResults/nbeats/top1_08098.yaml](config/high10mResults/nbeats/top1_08098.yaml)
2. [config/high10mResults/ulr/top1_04212.yaml](config/high10mResults/ulr/top1_04212.yaml)
3. [config/high30mResults/nbeats/top1_08745.yaml](config/high30mResults/nbeats/top1_08745.yaml)
4. [config/high1hResults/nbeats/top1_06725.yaml](config/high1hResults/nbeats/top1_06725.yaml)
5. [config/high1hResults/ulr/top1_03019.yaml](config/high1hResults/ulr/top1_03019.yaml)

There is no `top1` file for `30m / ulr` in the current repository.

## What The Models Actually Expect Internally

From TSMM training and evaluation code:

1. `ULR` uses flattened windows of shape `(batch_size, n_steps * n_features)` at inference.
2. `NBEATS` also ends up consuming flattened windows of shape `(batch_size, n_steps * n_features)` in this repo.

So the eDep-facing object schema should wrap one of these per sample:

1. matrix-style window, shape `(n_steps, n_features)`
2. flat vector, shape `(n_steps * n_features,)`

Matrix-style is clearer and safer.

## Batch Input Summary

| Timeframe | Model | n_steps | n_features | Preferred Object Field Shape | Flattened Object Field Shape |
|---|---|---:|---:|---|---|
| 10m | nbeats | 72 | 4 | `window: (72, 4)` | `vector: (288,)` |
| 10m | ulr | 72 | 3 | `window: (72, 3)` | `vector: (216,)` |
| 30m | nbeats | 42 | 5 | `window: (42, 5)` | `vector: (210,)` |
| 1h | nbeats | 3 | 5 | `window: (3, 5)` | `vector: (15,)` |
| 1h | ulr | 3 | 4 | `window: (3, 4)` | `vector: (12,)` |

## Feature Order Per Model

Feature order matters.

### 10m NBEATS

1. `HIGH`
2. `y_diff`
3. `Low_return`
4. `Price_return`

### 10m ULR

1. `HIGH`
2. `Price_return`
3. `y_diff`

### 30m NBEATS

1. `HIGH`
2. `y_diff`
3. `Low_return`
4. `Price_return`
5. `Open_return`

### 1h NBEATS

1. `HIGH`
2. `y_diff`
3. `Price_return`
4. `Open_return`
5. `Low_return`

### 1h ULR

1. `HIGH`
2. `y_diff`
3. `Low_return`
4. `Price_return`

## eDep Input Schema Format

Because eDep wants an object schema with `properties`, use one of these patterns.

### Option A: Matrix-style object schema

Recommended.

Example for `10m / nbeats`:

```json
{
  "type": "object",
  "properties": {
    "window": {
      "type": "array",
      "items": {
        "type": "array",
        "items": {
          "type": "number"
        },
        "minItems": 4,
        "maxItems": 4
      },
      "minItems": 72,
      "maxItems": 72
    }
  },
  "required": ["window"]
}
```

The POST body then becomes:

```json
[
  {
    "window": [
      [2100.1, 0.2, -0.01, 0.03],
      [2100.3, 0.2, 0.00, 0.02]
    ]
  }
]
```

### Option B: Flattened vector object schema

Use this only if you want the client to flatten before sending.

```json
{
  "type": "object",
  "properties": {
    "vector": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "minItems": 288,
      "maxItems": 288
    }
  },
  "required": ["vector"]
}
```

The POST body then becomes:

```json
[
  {
    "vector": [2100.1, 0.2, -0.01, 0.03]
  }
]
```

## Exact eDep Schemas Per Available `top1`

### 1. 10m NBEATS

Source: [config/high10mResults/nbeats/top1_08098.yaml](config/high10mResults/nbeats/top1_08098.yaml)

1. `n_steps = 72`
2. `n_features = 4`
3. feature order = `[HIGH, y_diff, Low_return, Price_return]`
4. preferred object payload = `{"window": [[... 72 rows ...]]}`
5. flat object payload = `{"vector": [... 288 values ...]}`

Matrix schema:

```json
{
  "type": "object",
  "properties": {
    "window": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" },
        "minItems": 4,
        "maxItems": 4
      },
      "minItems": 72,
      "maxItems": 72
    }
  },
  "required": ["window"]
}
```

### 2. 10m ULR

Source: [config/high10mResults/ulr/top1_04212.yaml](config/high10mResults/ulr/top1_04212.yaml)

1. `n_steps = 72`
2. `n_features = 3`
3. feature order = `[HIGH, Price_return, y_diff]`
4. preferred object payload = `{"window": [[... 72 rows ...]]}`
5. flat object payload = `{"vector": [... 216 values ...]}`

Matrix schema:

```json
{
  "type": "object",
  "properties": {
    "window": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" },
        "minItems": 3,
        "maxItems": 3
      },
      "minItems": 72,
      "maxItems": 72
    }
  },
  "required": ["window"]
}
```

### 3. 30m NBEATS

Source: [config/high30mResults/nbeats/top1_08745.yaml](config/high30mResults/nbeats/top1_08745.yaml)

1. `n_steps = 42`
2. `n_features = 5`
3. feature order = `[HIGH, y_diff, Low_return, Price_return, Open_return]`
4. preferred object payload = `{"window": [[... 42 rows ...]]}`
5. flat object payload = `{"vector": [... 210 values ...]}`

Matrix schema:

```json
{
  "type": "object",
  "properties": {
    "window": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" },
        "minItems": 5,
        "maxItems": 5
      },
      "minItems": 42,
      "maxItems": 42
    }
  },
  "required": ["window"]
}
```

### 4. 1h NBEATS

Source: [config/high1hResults/nbeats/top1_06725.yaml](config/high1hResults/nbeats/top1_06725.yaml)

1. `n_steps = 3`
2. `n_features = 5`
3. feature order = `[HIGH, y_diff, Price_return, Open_return, Low_return]`
4. preferred object payload = `{"window": [[... 3 rows ...]]}`
5. flat object payload = `{"vector": [... 15 values ...]}`

Matrix schema:

```json
{
  "type": "object",
  "properties": {
    "window": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" },
        "minItems": 5,
        "maxItems": 5
      },
      "minItems": 3,
      "maxItems": 3
    }
  },
  "required": ["window"]
}
```

### 5. 1h ULR

Source: [config/high1hResults/ulr/top1_03019.yaml](config/high1hResults/ulr/top1_03019.yaml)

1. `n_steps = 3`
2. `n_features = 4`
3. feature order = `[HIGH, y_diff, Low_return, Price_return]`
4. preferred object payload = `{"window": [[... 3 rows ...]]}`
5. flat object payload = `{"vector": [... 12 values ...]}`

Matrix schema:

```json
{
  "type": "object",
  "properties": {
    "window": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" },
        "minItems": 4,
        "maxItems": 4
      },
      "minItems": 3,
      "maxItems": 3
    }
  },
  "required": ["window"]
}
```

## Shared Ingestion Pattern For eDep

With eDep, the shared serving pattern is now:

1. Receive `list[dict]` from the generated FastAPI route.
2. Read `window` or `vector` from each object.
3. Convert to `numpy`.
4. If the payload is matrix-style, flatten it to `(batch, n_steps * n_features)`.
5. Apply `scaler_X.transform(...)`.
6. Run model inference.
7. Apply `scaler_y.inverse_transform(...)`.
8. Convert first-step `y_diff` into `forecast_sign`, `signal`, and `confidence`.

## eDep-Ready `ingestion_pipeline.py`

Use the standalone template here:

1. [scripts/edep_ingestion_pipeline.py](scripts/edep_ingestion_pipeline.py)

That template expects a single bundled `.pkl` model file exported with:

1. [scripts/export_edep_bundle.py](scripts/export_edep_bundle.py)

## Recommended eDep Workflow

1. Export one TSMM predictor bundle with [scripts/export_edep_bundle.py](scripts/export_edep_bundle.py).
2. Upload that `.pkl` bundle as the model file in eDep.
3. Upload [scripts/edep_ingestion_pipeline.py](scripts/edep_ingestion_pipeline.py) as `ingestion_pipeline.py`.
4. Paste the matching object-style `input_schema.json` from this document into eDep.
5. Use POST body shape `[{"window": ...}]` for inference.

## Recommendation

For eDep specifically, use matrix-style object payloads:

1. They match eDep’s object-schema generator.
2. They make feature ordering visible.
3. They keep the flattening inside the ingestion layer.
4. One shared ingestion pipeline works for all 5 currently available `top1` models.
5. The bundle exporter resolves eDep’s single-model-file limitation.
