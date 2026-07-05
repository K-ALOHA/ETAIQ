# ETAIQ Metrics Persistence Audit Report

**Date**: 2026-07-03  
**Scope**: All 8 key metrics tracking their calculation, persistence, and API exposure

---

## Executive Summary

| Metric | Calculated | Persisted | Backend API | Frontend | Overall Status |
|--------|:----------:|:---------:|:-----------:|:--------:|:---------------:|
| MAE | ✅ | ✅ | ✅ | ⚠️ | 🟢 Good |
| RMSE | ✅ | ✅ | ❌ | ⚠️ | 🟠 Incomplete |
| R² | ✅ | ⚠️ | ❌ | ⚠️ | 🟠 Incomplete |
| Training Time | ✅ | ✅ | ✅ | ✅ | 🟢 Good |
| Latency | ✅ | ❌ | ⚠️ | ❌ | 🔴 Missing |
| Dataset Size | ✅ | ✅ | ❌ | ⚠️ | 🟠 Incomplete |
| Training Samples | ✅ | ⚠️ | ❌ | ⚠️ | 🟠 Incomplete |
| Model Status | ✅ | ✅ | ⚠️ | ✅ | 🟠 Incomplete |

---

## 1. MAE (Mean Absolute Error)

### Is it calculated?
✅ **YES** 
- **File**: [ml/training/evaluation.py](ml/training/evaluation.py#L71)
- **Code**: Line 71
  ```python
  mae = float(mean_absolute_error(y_array, predictions))
  ```

### Where persisted?

✅ **MULTIPLE SOURCES:**

1. **Registry JSON files** (Primary source)
   - **Location**: [ml/data/training/model_registry/](ml/data/training/model_registry/)
   - **Pattern**: `{ModelName}_v{version}_registry.json`
   - **Field**: `metrics.mae`
   - **Example 1**: [LinearRegression_v1038_registry.json](ml/data/training/model_registry/LinearRegression_v1038_registry.json) line 6
     ```json
     "metrics": {
       "mae": 8.881784197001252e-16
     }
     ```
   - **Example 2**: [XGBRegressor_v1_registry.json](ml/data/training/model_registry/XGBRegressor_v1_registry.json) line 6
     ```json
     "metrics": {
       "mae": 4.437311712612657
     }
     ```

2. **Experiments JSON** (Audit trail)
   - **Location**: [ml/data/training/experiments/experiments.json](ml/data/training/experiments/experiments.json)
   - **Field**: `metrics.mae`
   - **Example**: Line 9
     ```json
     "metrics": {
       "mae": 0.0
     }
     ```
   - **Persistence mechanism**: [ml/training/experiment_tracking.py](ml/training/experiment_tracking.py) line 81-82

### Is it loaded by backend API?

✅ **YES**
- **Endpoint**: `GET /api/v1/models`
- **File**: [backend/app/api/prediction.py](backend/app/api/prediction.py) line 179-190
- **Mechanism**: Loads from registry via `registry_engine.list_models()`
- **Format**: Included in registry model metadata

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Calculation** | [ml/training/evaluation.py:71](ml/training/evaluation.py#L71) - EvaluationEngine.evaluate() |
| **Persistence** | [ml/training/model_registry.py:155-160](ml/training/model_registry.py#L155-L160) - ModelRegistryEngine._persist_registration() |
| **Loading** | [ml/training/model_registry.py](ml/training/model_registry.py) - loads from JSON at startup |

### Conclusion
🟢 **GOOD** - MAE is properly calculated, persisted, and exposed through the backend API.

---

## 2. RMSE (Root Mean Squared Error)

### Is it calculated?
✅ **YES**
- **File**: [ml/training/evaluation.py](ml/training/evaluation.py#L72)
- **Code**: Line 72
  ```python
  rmse = float(np.sqrt(mean_squared_error(y_array, predictions)))
  ```

### Where persisted?

⚠️ **PARTIALLY PERSISTED** - Only in some models

1. **Registry JSON files** (Inconsistent)
   - **Location**: [ml/data/training/model_registry/](ml/data/training/model_registry/)
   - **Field**: `metrics.rmse`
   - **Status**: ONLY in XGBRegressor and GradientBoostingRegressor models
   - **Example**: [XGBRegressor_v1_registry.json](ml/data/training/model_registry/XGBRegressor_v1_registry.json) line 7
     ```json
     "metrics": {
       "rmse": 7.51258761808612
     }
     ```
   - **Missing from**: LinearRegression models (all versions)
     - [LinearRegression_v1038_registry.json](ml/data/training/model_registry/LinearRegression_v1038_registry.json) - NO rmse field

2. **Cross-validation results** (Calculated but not exported)
   - **File**: [ml/training/cross_validation.py](ml/training/cross_validation.py) line 29-32
   - **Structure**: CrossValidationResult includes `mean_rmse` and `std_rmse`
   - **Export path**: Line 47 - `_export_path = DEFAULT_TRAINING_CONFIG.models_dir / "cross_validation_results.csv"`
   - **Issue**: Results are calculated but CSV export not actively used

### Is it loaded by backend API?

❌ **NO** - RMSE is NOT exposed in API responses

- **Problem**: [backend/app/api/training.py](backend/app/api/training.py) line 36 only returns MAE
  ```python
  metrics={"mae": float(result.best_model.metric_value)},
  ```
- **Backend Response**: [backend/app/schemas/ml_management.py](backend/app/schemas/ml_management.py) - TrainingResponse only includes MAE

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Calculation** | [ml/training/evaluation.py:72](ml/training/evaluation.py#L72) |
| **Persistence** | [ml/training/model_registry.py:155-160](ml/training/model_registry.py#L155-L160) (inconsistent) |
| **Gap** | ❌ Not exposed through backend API |

### Conclusion
🟠 **INCOMPLETE** - RMSE is calculated and stored in registry JSON for some models, but:
1. Not consistently persisted (missing from LinearRegression models)
2. Not exposed through backend API endpoints
3. Frontend uses hardcoded values: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx) line 31
   ```typescript
   { model: "LinearRegression", MAE: 6.51, RMSE: 8.31, R2: 0.22 }
   ```

### Recommendation
🔴 **ACTION REQUIRED**:
- Add RMSE to all model registry entries
- Include RMSE in `ModelInfoResponse` schema
- Update backend `/models` endpoint to return RMSE
- Remove hardcoded values from frontend

---

## 3. R² (R-squared / Coefficient of Determination)

### Is it calculated?
✅ **YES**
- **File**: [ml/training/evaluation.py](ml/training/evaluation.py#L73)
- **Code**: Line 73
  ```python
  r2 = float(r2_score(y_array, predictions))
  ```

### Where persisted?

⚠️ **INCONSISTENTLY PERSISTED** - Only in some models

1. **Registry JSON files** (Selective)
   - **Location**: [ml/data/training/model_registry/](ml/data/training/model_registry/)
   - **Field**: `metrics.r2`
   - **Present in**: XGBRegressor_v1, XGBRegressor_v2, GradientBoostingRegressor models
   - **Example**: [XGBRegressor_v1_registry.json](ml/data/training/model_registry/XGBRegressor_v1_registry.json) line 8
     ```json
     "metrics": {
       "r2": 0.37543886965059114
     }
     ```
   - **Missing from**: LinearRegression models (all versions)
     - [LinearRegression_v1038_registry.json](ml/data/training/model_registry/LinearRegression_v1038_registry.json) - ONLY has MAE

2. **Cross-validation results** (Tracked but not exported)
   - **File**: [ml/training/cross_validation.py](ml/training/cross_validation.py) line 33-34
   - **Structure**: CrossValidationResult includes `mean_r2` and `std_r2`

### Is it loaded by backend API?

❌ **NO** - R² is NOT exposed in backend API

- **Problem**: Not included in any API response schema
- **Frontend workaround**: Uses hardcoded values: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx) line 31

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Calculation** | [ml/training/evaluation.py:73](ml/training/evaluation.py#L73) |
| **Persistence** | [ml/training/model_registry.py:155-160](ml/training/model_registry.py#L155-L160) (selective) |
| **Gap** | ❌ Not exposed through backend API |

### Conclusion
🟠 **INCOMPLETE** - R² is calculated but:
1. Not consistently persisted (missing from LinearRegression and RandomForest models)
2. Not exposed through backend API at all
3. Frontend displays hardcoded values instead of real metrics

### Recommendation
🔴 **ACTION REQUIRED**:
- Ensure R² is calculated and stored for ALL models
- Add R² to `ModelInfoResponse` schema
- Update backend `/models` endpoint
- Implement version history tracking

---

## 4. Training Time

### Is it calculated?
✅ **YES** - Tracked in multiple places

1. **TrainingEngine**
   - **File**: [ml/training/trainer.py](ml/training/trainer.py) line 50-65
   - **Code**:
     ```python
     start_time = time.perf_counter()
     # ... training logic ...
     elapsed = time.perf_counter() - start_time
     ```
   - **Returns**: TrainingResult with `training_time_seconds`

2. **TrainingService**
   - **File**: [ml/training/training_service.py](ml/training/training_service.py) line 43
   - **Aggregates** total time across all models

### Where persisted?

✅ **MULTIPLE LOCATIONS:**

1. **Experiments JSON** (Primary audit trail)
   - **Location**: [ml/data/training/experiments/experiments.json](ml/data/training/experiments/experiments.json)
   - **Field**: `training_time_seconds`
   - **Example**: Line 11
     ```json
     "training_time_seconds": 0.01
     ```
   - **Persisted by**: [ml/training/experiment_tracking.py](ml/training/experiment_tracking.py) line 81-82

2. **Registry metadata** (Timestamp)
   - **Location**: [ml/data/training/model_registry/](ml/data/training/model_registry/)
   - **Field**: `metadata.training_timestamp`
   - **Example**: [XGBRegressor_v1_registry.json](ml/data/training/model_registry/XGBRegressor_v1_registry.json) line 38
     ```json
     "metadata": {
       "training_timestamp": "2026-07-03T03:51:38Z"
     }
     ```

### Is it loaded by backend API?

✅ **YES** - Fully exposed

- **Endpoint**: `POST /api/v1/train` returns training_time_seconds
- **File**: [backend/app/api/training.py](backend/app/api/training.py) line 48
  ```python
  training_time_seconds=result.training_time
  ```
- **Schema**: [backend/app/schemas/ml_management.py](backend/app/schemas/ml_management.py) line 37
  ```python
  training_time_seconds: float = Field(description="Total training elapsed time")
  ```

### Frontend usage?

✅ **YES** - Dashboard fetches and displays

- **File**: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx)
- **Fetches from**: `/api/v1/experiments`
- **Line 224**: Displays `record.training_time_seconds`

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Calculation** | [ml/training/trainer.py:55-65](ml/training/trainer.py#L55-L65) |
| **Persistence** | [ml/training/experiment_tracking.py:81-82](ml/training/experiment_tracking.py#L81-L82) |
| **Loading** | Backend API and `/experiments` endpoint |

### Conclusion
🟢 **GOOD** - Training time is:
- ✅ Properly calculated
- ✅ Persistently stored
- ✅ Exposed through backend API
- ✅ Displayed in frontend with real data

---

## 5. Latency (Prediction Latency / P99 Response Time)

### Is it calculated?
✅ **YES** - Calculated per prediction

- **File**: [backend/app/api/prediction.py](backend/app/api/prediction.py) line 155-161
- **Code**:
  ```python
  start_time = time.perf_counter()
  # ... prediction logic ...
  elapsed_ms = (time.perf_counter() - start_time) * 1000.0
  ```
- **Returned in**: Every prediction response

### Where persisted?

❌ **NOT PERSISTED** - Only in memory

- **Issue**: Latency is calculated but never saved to disk
- **Problem**: No historical data, cannot calculate P99/P95 percentiles
- **Consequence**: SLA monitoring impossible

### Is it loaded by backend API?

⚠️ **PARTIAL** - Returned in response only, not accessible for analysis

- **Endpoint**: `POST /api/v1/predict` returns in response
- **File**: [backend/app/schemas/prediction.py](backend/app/schemas/prediction.py) line 46
  ```python
  processing_time_ms: float = Field(description="Processing duration in milliseconds")
  ```
- **Response example**: Each prediction includes `processing_time_ms: float`

### Frontend usage?

❌ **NO** - Not fetched from backend

- **Issue**: Frontend displays hardcoded value: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx) line 81
  ```typescript
  { title: "Prediction Latency", subtitle: "API response time", trendLabel: "99th %ile" }
  ```
- **Shows**: Hardcoded "99th %ile" instead of real P99 metric

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Calculation** | [backend/app/api/prediction.py:155-161](backend/app/api/prediction.py#L155-L161) |
| **Persistence** | ❌ MISSING |
| **Analysis** | ❌ MISSING - No P99/P95 aggregation |

### Conclusion
🔴 **CRITICAL GAP** - Latency is calculated but:
1. ❌ Not persisted to any database
2. ❌ No aggregation for P99, P95, mean, etc.
3. ❌ No monitoring endpoint to retrieve historical latency
4. ❌ No SLA tracking capability
5. Frontend shows "99th %ile" as hardcoded string, not real metric

### Recommendation
🔴 **ACTION REQUIRED** (High Priority):
1. Add metrics table to database (or use Redis/TimescaleDB)
2. Store each prediction latency with timestamp
3. Implement `/monitoring/latency` endpoint with aggregations:
   - Mean latency (sliding window)
   - P50, P95, P99 percentiles
   - Latency by model version
4. Alert if P99 exceeds 200ms (per PROJECT_SPEC.md)
5. Update frontend to fetch real latency metrics

---

## 6. Dataset Size (Orders / Records)

### Is it calculated?
✅ **YES** - Multiple locations count records

1. **DataSplitEngine** (Most important)
   - **File**: [ml/features/data_split.py](ml/features/data_split.py) line 45-49
   - **Code**: Logs total rows
   - **Method**: `len(engineered_dataset)`, `len(X_train)`, `len(X_test)`

2. **TrainingEngine**
   - **File**: [ml/training/trainer.py](ml/training/trainer.py) line 48-49
   - **Tracks**: `training_rows=len(np.asarray(X_train))`

### Where persisted?

✅ **REGISTRY JSON FILES**

- **Location**: [ml/data/training/model_registry/](ml/data/training/model_registry/)
- **Field**: `metadata.dataset_size`
- **Value**: Total number of records used during training
- **Example**: [XGBRegressor_v1_registry.json](ml/data/training/model_registry/XGBRegressor_v1_registry.json) line 14
  ```json
  "metadata": {
    "dataset_size": 264777,
    "feature_count": 14
  }
  ```

### Is it loaded by backend API?

❌ **NO** - Dataset size NOT exposed

- **Problem**: Registry data is loaded but `ModelInfoResponse` doesn't include dataset_size
- **File**: [backend/app/api/prediction.py](backend/app/api/prediction.py) line 179-190 loads registry but doesn't return all metadata

### Frontend usage?

❌ **NO** - Frontend displays hardcoded value

- **File**: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx) line 68
- **Shows**: Hardcoded "Dataset Size" card instead of real value
- **Backend data NOT fetched**

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Calculation** | [ml/features/data_split.py:45-49](ml/features/data_split.py#L45-L49) |
| **Persistence** | [ml/data/training/model_registry/](ml/data/training/model_registry/) - metadata.dataset_size |
| **Gap** | ❌ Not exposed through backend API |

### Conclusion
🟠 **INCOMPLETE** - Dataset size is:
1. ✅ Calculated and stored in registry
2. ❌ NOT exposed through backend API
3. ❌ Frontend uses hardcoded value

### Recommendation
🔴 **ACTION REQUIRED**:
1. Include dataset_size in `ModelInfoResponse`
2. Update `/models` endpoint to return dataset metadata
3. Update frontend to fetch real value from backend
4. Add feature_count and other metadata too

---

## 7. Training Samples (Records in training set)

### Is it calculated?
✅ **YES** - Calculated at split time

1. **DataSplitEngine** (Primary)
   - **File**: [ml/features/data_split.py](ml/features/data_split.py) line 45-49
   - **Logs**: `training_rows=len(X_train)`

2. **TrainingEngine**
   - **File**: [ml/training/trainer.py](ml/training/trainer.py) line 48-49
   - **Logs**: `training_rows=len(np.asarray(X_train))`

### Where persisted?

⚠️ **PARTIALLY PERSISTED** - Not consistently saved

1. **TrainingResult** (Runtime only)
   - **File**: [ml/training/trainer.py](ml/training/trainer.py) line 27
   - **Field**: `training_rows: int`
   - **Issue**: Only in memory, not persisted to disk

2. **Experiment records** (NOT tracked)
   - **File**: [ml/training/experiment_tracking.py](ml/training/experiment_tracking.py) line 26-28
   - **Issue**: ExperimentRecord does NOT include training_samples field
   - **Could be**: Added to hyperparameters dict, but not currently

3. **Registry metadata** (Not included)
   - **File**: [ml/data/training/model_registry/](ml/data/training/model_registry/)
   - **Issue**: Registry includes dataset_size and feature_count, but not training_samples
   - **Example**: [XGBRegressor_v1_registry.json](ml/data/training/model_registry/XGBRegressor_v1_registry.json) - no training_samples field

### Is it loaded by backend API?

❌ **NO** - Training samples NOT available through API

- **Problem**: Not stored in any backend-accessible location
- **Consequence**: Cannot calculate train/test split ratio (80/20)

### Frontend usage?

❌ **NO** - Frontend displays hardcoded value

- **File**: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx) line 72
- **Shows**: Hardcoded "Training Samples" card

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Calculation** | [ml/features/data_split.py:45-49](ml/features/data_split.py#L45-L49) |
| **Persistence** | ⚠️ Runtime only (TrainingResult object) |
| **Gap** | ❌ Not persisted to disk |

### Conclusion
🟠 **INCOMPLETE** - Training samples:
1. ✅ Calculated at split time
2. ❌ Not persisted to disk (only in memory)
3. ❌ Not accessible through backend API
4. ❌ Frontend displays hardcoded value

### Recommendation
🔴 **ACTION REQUIRED**:
1. Add training_samples to ExperimentRecord
2. Persist to experiments.json
3. Include in registry metadata (or calculate from dataset_size * 0.8)
4. Expose through `/models` endpoint
5. Update frontend to fetch real values

---

## 8. Model Status (Health / Availability)

### Is it calculated?
✅ **YES** - Status assigned at registration

- **File**: [ml/training/model_registry.py](ml/training/model_registry.py) line 62
- **Field**: `status: str` in RegisteredModel dataclass
- **Assigned at**: Registration time
- **Possible values**: "Production", "Archived", "Candidate"

### Where persisted?

✅ **REGISTRY JSON FILES**

- **Location**: [ml/data/training/model_registry/](ml/data/training/model_registry/)
- **Field**: `"status"` at root level
- **Example**: [XGBRegressor_v1_registry.json](ml/data/training/model_registry/XGBRegressor_v1_registry.json) line 10
  ```json
  "status": "Production"
  ```
- **Persisted by**: [ml/training/model_registry.py:155-160](ml/training/model_registry.py#L155-L160) (_persist_registration)

### Is it loaded by backend API?

⚠️ **PARTIAL** - Loaded but not fully utilized

1. **Health endpoint** (Hardcoded)
   - **File**: [backend/app/api/health.py](backend/app/api/health.py) line 15-22
   - **Returns**: `status: "healthy"` (hardcoded)
   - **Problem**: Does NOT check if models are actually loaded
   - **Code**:
     ```python
     return HealthResponse(status="healthy")
     ```

2. **Prediction endpoint** (Uses status correctly)
   - **File**: [backend/app/api/prediction.py](backend/app/api/prediction.py) line 39-46
   - **Filters**: `model.status == "Production"`
   - **Correct usage**: Selects only production models

3. **Models endpoint** (Lists models)
   - **File**: [backend/app/api/prediction.py](backend/app/api/prediction.py) line 200
   - **Loads**: Production models via registry

### Frontend usage?

✅ **YES** - Displays model status

- **File**: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx)
- **Fetches from**: `/api/v1/models` endpoint
- **Status shown**: In model comparison data

### Single source of truth?

| Component | Location |
|-----------|----------|
| **Assignment** | [ml/training/model_registry.py:62](ml/training/model_registry.py#L62) |
| **Persistence** | [ml/training/model_registry.py:155-160](ml/training/model_registry.py#L155-L160) |
| **Loading** | Backend API loads from registry |
| **Gap** | Health endpoint returns hardcoded "healthy" |

### Conclusion
🟠 **INCOMPLETE** - Model status:
1. ✅ Properly calculated and persisted
2. ✅ Used correctly in prediction endpoint
3. ⚠️ Health endpoint returns hardcoded "healthy"
4. ✅ Models endpoint lists status correctly

### Recommendation
🟠 **SHOULD FIX**:
1. Update `/health` endpoint to check actual model availability
2. Instead of hardcoded "healthy", check:
   - Is production model loaded?
   - Can registry be accessed?
   - Are all required dependencies available?

---

## Summary Table

| Metric | Calculated | Persisted | Backend Exposed | Frontend Uses Real Data | Overall |
|--------|:----------:|:---------:|:---------------:|:----------------------:|:--------:|
| **MAE** | ✅ Yes | ✅ Registry JSON | ✅ Yes | ❌ Hardcoded | 🟢 Good |
| **RMSE** | ✅ Yes | ⚠️ Selective | ❌ No | ❌ Hardcoded | 🟠 Incomplete |
| **R²** | ✅ Yes | ⚠️ Selective | ❌ No | ❌ Hardcoded | 🟠 Incomplete |
| **Training Time** | ✅ Yes | ✅ experiments.json | ✅ Yes | ✅ Yes | 🟢 Good |
| **Latency** | ✅ Yes | ❌ No | ⚠️ Per-response | ❌ Hardcoded | 🔴 Critical Gap |
| **Dataset Size** | ✅ Yes | ✅ Registry JSON | ❌ No | ❌ Hardcoded | 🟠 Incomplete |
| **Training Samples** | ✅ Yes | ❌ Runtime only | ❌ No | ❌ Hardcoded | 🟠 Incomplete |
| **Model Status** | ✅ Yes | ✅ Registry JSON | ⚠️ Partial | ✅ Yes | 🟠 Incomplete |

---

## Critical Issues Summary

### 🔴 CRITICAL ISSUES (Production Blockers)

#### Issue 1: Latency metrics not persisted
- **Impact**: Cannot track P99 latency for SLA compliance
- **Requirement** (per PROJECT_SPEC.md line 56): "Latency: Prediction API p99 < 200ms"
- **Current state**: Latency calculated but never saved
- **Fix effort**: Medium (need metrics DB/table)
- **Priority**: 🔴 CRITICAL

#### Issue 2: Backend API doesn't expose model metrics
- **Affected metrics**: RMSE, R², dataset_size, training_samples
- **Current state**: Hardcoded in frontend instead of fetched from backend
- **Problem**: Cannot track actual model performance
- **Fix effort**: Low (schema + endpoint update)
- **Priority**: 🔴 CRITICAL

---

### 🟠 HIGH PRIORITY ISSUES

#### Issue 3: RMSE not consistently saved
- **Affected models**: LinearRegression (all versions) missing RMSE
- **Example gap**: [LinearRegression_v1038_registry.json](ml/data/training/model_registry/LinearRegression_v1038_registry.json) - only MAE
- **Fix effort**: Low (add to evaluation store)
- **Priority**: 🟠 HIGH

#### Issue 4: R² not consistently saved
- **Affected models**: LinearRegression, RandomForest missing R²
- **Fix effort**: Low (add to evaluation store)
- **Priority**: 🟠 HIGH

#### Issue 5: Training samples not persisted
- **Current state**: Only in memory during training
- **Loss**: Cannot reconstruct train/test split after training
- **Fix effort**: Low (add to experiment_tracking.py)
- **Priority**: 🟠 HIGH

---

### 🟠 MEDIUM PRIORITY ISSUES

#### Issue 6: Health endpoint returns hardcoded status
- **Current**: Always returns "healthy"
- **Should**: Check actual model availability
- **File**: [backend/app/api/health.py](backend/app/api/health.py)
- **Fix effort**: Low
- **Priority**: 🟠 MEDIUM

---

## Recommended Fix Priority

### Phase 1 (Immediate - Production Blocker)
1. **Add latency persistence** 
   - Store prediction latencies with timestamp
   - Add P99 aggregation endpoint
   - Update frontend to fetch real metrics

2. **Expose all metrics in backend API**
   - Update ModelInfoResponse to include RMSE, R², dataset_size, training_samples
   - Update `/models` endpoint response
   - Update `/train` endpoint response

### Phase 2 (High Priority)
3. **Ensure all metrics are consistently calculated and stored**
   - Add missing RMSE to LinearRegression models
   - Add missing R² to LinearRegression/RandomForest models
   - Add training_samples to experiments.json

4. **Update frontend to fetch real data**
   - Remove hardcoded KPI values
   - Fetch from `/models` endpoint
   - Fetch from `/experiments` endpoint

### Phase 3 (Medium Priority)
5. **Improve health endpoint**
   - Check actual model availability
   - Check registry accessibility

---

## File Reference Map

### ML Training & Evaluation
- **Calculation**: [ml/training/evaluation.py](ml/training/evaluation.py) - EvaluationEngine
- **Persistence**: [ml/training/model_registry.py](ml/training/model_registry.py) - ModelRegistryEngine
- **Experiments**: [ml/training/experiment_tracking.py](ml/training/experiment_tracking.py) - ExperimentTrackingEngine
- **Data split**: [ml/features/data_split.py](ml/features/data_split.py) - DataSplitEngine

### Backend API
- **Prediction API**: [backend/app/api/prediction.py](backend/app/api/prediction.py)
- **Training API**: [backend/app/api/training.py](backend/app/api/training.py)
- **Health API**: [backend/app/api/health.py](backend/app/api/health.py)
- **Schemas**: [backend/app/schemas/](backend/app/schemas/)

### Frontend Dashboard
- **Dashboard**: [frontend/src/app/(app)/dashboard/page.tsx](frontend/src/app/(app)/dashboard/page.tsx)
- **API client**: [frontend/src/lib/dashboard.ts](frontend/src/lib/dashboard.ts)

### Data Persistence
- **Registry**: [ml/data/training/model_registry/](ml/data/training/model_registry/) (25+ JSON files)
- **Experiments**: [ml/data/training/experiments/experiments.json](ml/data/training/experiments/experiments.json)

---

## Conclusion

**Overall Assessment**: 🟠 **Partially production-ready with critical gaps**

**Strengths**:
- ✅ MAE properly calculated and exposed
- ✅ Training time tracked end-to-end
- ✅ Model status persisted and used correctly
- ✅ Multiple data persistence layers

**Critical Gaps**:
- 🔴 Latency metrics not persisted (SLA compliance impossible)
- 🔴 Backend API missing RMSE, R², dataset_size, training_samples
- 🔴 Frontend displays hardcoded values instead of real metrics
- 🔴 Health endpoint returns hardcoded status

**To reach production-ready**:
1. Implement latency metrics persistence (P99 tracking)
2. Expose all metrics through backend API
3. Update frontend to fetch real data
4. Add missing metrics to consistent models

**Estimated effort**: 2-3 days for all fixes
