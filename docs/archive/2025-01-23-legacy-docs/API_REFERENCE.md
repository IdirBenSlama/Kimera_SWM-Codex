# Kimera SWM API Reference

## Overview

This document provides a comprehensive reference for the Kimera SWM API endpoints. The API follows RESTful principles and uses JSON for data exchange.

## Base URL

```
http://localhost:8000
```

## Authentication

Authentication is required for all endpoints except `/health` and `/metrics`. The API uses JWT (JSON Web Token) authentication.

### Authentication Headers

```
Authorization: Bearer <token>
```

## Common Response Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request - Invalid input parameters |
| 401  | Unauthorized - Authentication required |
| 403  | Forbidden - Insufficient permissions |
| 404  | Not Found - Resource not found |
| 500  | Internal Server Error |

## Endpoints

### Health Check

#### GET /health

Check the health status of the API.

**Response:**

```json
{
  "status": "healthy",
  "database_connection": true,
  "components": {
    "vault_manager": true,
    "embedding_model": true,
    "contradiction_engine": true,
    "thermodynamic_engine": true,
    "spde_engine": true,
    "cognitive_cycle_engine": true,
    "meta_insight_engine": true,
    "proactive_detector": true,
    "revolutionary_intelligence_engine": true,
    "geoid_scar_manager": true,
    "system_monitor": true,
    "ethical_governor": true
  },
  "version": "0.1.0"
}
```

### System Status

#### GET /kimera/status

Get detailed system status information.

**Response:**

```json
{
  "status": "operational",
  "uptime_seconds": 3600,
  "database": {
    "type": "PostgreSQL",
    "version": "15.12",
    "connection_status": "connected",
    "pgvector_available": true
  },
  "scientific_components": {
    "thermodynamic_engine": {
      "status": "operational",
      "entropy_value": 6.73,
      "error_margin": 1e-10
    },
    "quantum_field_engine": {
      "status": "operational",
      "coherence": 1.0
    },
    "spde_engine": {
      "status": "operational",
      "conservation_error": 0.0018
    },
    "portal_mechanics": {
      "status": "operational",
      "stability": 0.9872
    }
  },
  "memory_usage": {
    "total_mb": 1024,
    "used_mb": 512,
    "percent": 50.0
  },
  "gpu_status": {
    "available": true,
    "device": "NVIDIA GeForce RTX 2080 Ti",
    "memory_total_mb": 11264,
    "memory_used_mb": 2048,
    "utilization_percent": 18.2
  }
}
```

### Cognitive Field Operations

#### POST /kimera/cognitive/field

Process a cognitive field through the system.

**Request:**

```json
{
  "field": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
  "expected_dimensions": 3,
  "evolution_steps": 5,
  "temperature": 0.5
}
```

**Response:**

```json
{
  "field_id": "f8c3de3d-1fea-4d7c-a8b0-29f63c4c3454",
  "processed_field": [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]],
  "entropy": 1.37,
  "coherence_factor": 0.92,
  "energy_level": 3.45,
  "processing_time_ms": 42
}
```

### Geoid Operations

#### POST /kimera/geoid

Create a new geoid.

**Request:**

```json
{
  "symbolic_state": {
    "concept_a": 0.8,
    "concept_b": 0.6,
    "concept_c": 0.9
  },
  "metadata": {
    "source": "user_input",
    "context": "exploration"
  },
  "semantic_state": "This is a semantic representation of the state."
}
```

**Response:**

```json
{
  "geoid_id": "a1b2c3d4-e5f6-4a5b-8c9d-1e2f3a4b5c6d",
  "entropy": 1.25,
  "coherence_factor": 0.87,
  "creation_timestamp": "2023-06-14T12:34:56.789Z"
}
```

#### GET /kimera/geoid/{geoid_id}

Retrieve a geoid by ID.

**Response:**

```json
{
  "geoid_id": "a1b2c3d4-e5f6-4a5b-8c9d-1e2f3a4b5c6d",
  "symbolic_state": {
    "concept_a": 0.8,
    "concept_b": 0.6,
    "concept_c": 0.9
  },
  "metadata": {
    "source": "user_input",
    "context": "exploration"
  },
  "semantic_state": "This is a semantic representation of the state.",
  "entropy": 1.25,
  "coherence_factor": 0.87,
  "energy_level": 2.5,
  "creation_timestamp": "2023-06-14T12:34:56.789Z"
}
```

### SCAR Operations

#### POST /kimera/scar

Create a new SCAR (State Change Analysis Record).

**Request:**

```json
{
  "source_geoid_id": "a1b2c3d4-e5f6-4a5b-8c9d-1e2f3a4b5c6d",
  "target_geoid_id": "b2c3d4e5-f6a7-5b6c-9d0e-2f3a4b5c6d7e",
  "transition_type": "cognitive_leap",
  "metadata": {
    "trigger": "contradiction_resolution",
    "context": "problem_solving"
  }
}
```

**Response:**

```json
{
  "scar_id": "c3d4e5f6-a7b8-6c7d-0e1f-3a4b5c6d7e8f",
  "transition_energy": 1.75,
  "conservation_error": 0.0012,
  "creation_timestamp": "2023-06-14T12:35:00.123Z"
}
```

#### GET /kimera/scar/{scar_id}

Retrieve a SCAR by ID.

**Response:**

```json
{
  "scar_id": "c3d4e5f6-a7b8-6c7d-0e1f-3a4b5c6d7e8f",
  "source_geoid_id": "a1b2c3d4-e5f6-4a5b-8c9d-1e2f3a4b5c6d",
  "target_geoid_id": "b2c3d4e5-f6a7-5b6c-9d0e-2f3a4b5c6d7e",
  "transition_type": "cognitive_leap",
  "transition_energy": 1.75,
  "conservation_error": 0.0012,
  "metadata": {
    "trigger": "contradiction_resolution",
    "context": "problem_solving"
  },
  "creation_timestamp": "2023-06-14T12:35:00.123Z"
}
```

### Contradiction Engine

#### POST /kimera/contradiction/analyze

Analyze contradictions between two statements or concepts.

**Request:**

```json
{
  "statement_a": "All swans are white.",
  "statement_b": "There exist black swans in Australia.",
  "context": "zoology"
}
```

**Response:**

```json
{
  "contradiction_detected": true,
  "contradiction_type": "existential_counterexample",
  "contradiction_strength": 0.95,
  "resolution_suggestions": [
    {
      "resolution_type": "qualification",
      "resolution_text": "Most swans in Europe are white, while some swans in Australia are black.",
      "confidence": 0.87
    },
    {
      "resolution_type": "definition_refinement",
      "resolution_text": "Clarify that 'swan' refers to multiple species with different colorations.",
      "confidence": 0.82
    }
  ],
  "entropy_delta": 1.45
}
```

### Vault Operations

#### POST /kimera/vault/store

Store data in the vault.

**Request:**

```json
{
  "data_type": "insight",
  "content": {
    "insight_type": "pattern_recognition",
    "description": "Recurring pattern identified in cognitive transitions",
    "confidence": 0.85,
    "supporting_evidence": ["geoid_id_1", "geoid_id_2", "geoid_id_3"]
  },
  "metadata": {
    "source": "meta_insight_engine",
    "priority": "high"
  }
}
```

**Response:**

```json
{
  "vault_entry_id": "d4e5f6a7-b8c9-7d8e-1f2a-4b5c6d7e8f9a",
  "storage_timestamp": "2023-06-14T12:36:12.456Z"
}
```

#### GET /kimera/vault/{entry_id}

Retrieve data from the vault by ID.

**Response:**

```json
{
  "vault_entry_id": "d4e5f6a7-b8c9-7d8e-1f2a-4b5c6d7e8f9a",
  "data_type": "insight",
  "content": {
    "insight_type": "pattern_recognition",
    "description": "Recurring pattern identified in cognitive transitions",
    "confidence": 0.85,
    "supporting_evidence": ["geoid_id_1", "geoid_id_2", "geoid_id_3"]
  },
  "metadata": {
    "source": "meta_insight_engine",
    "priority": "high"
  },
  "storage_timestamp": "2023-06-14T12:36:12.456Z",
  "last_accessed_timestamp": "2023-06-14T13:45:23.789Z",
  "access_count": 3
}
```

### Metrics

#### GET /metrics

Get Prometheus-compatible metrics for system monitoring.

**Response:**

```
# HELP kimera_geoid_count Total number of geoids in the system
# TYPE kimera_geoid_count gauge
kimera_geoid_count 1245

# HELP kimera_scar_count Total number of SCARs in the system
# TYPE kimera_scar_count gauge
kimera_scar_count 2134

# HELP kimera_average_entropy Average entropy across all geoids
# TYPE kimera_average_entropy gauge
kimera_average_entropy 2.34

# HELP kimera_database_connection_status Database connection status (1=connected, 0=disconnected)
# TYPE kimera_database_connection_status gauge
kimera_database_connection_status 1

# HELP kimera_api_requests_total Total number of API requests
# TYPE kimera_api_requests_total counter
kimera_api_requests_total{endpoint="/kimera/status"} 1532
kimera_api_requests_total{endpoint="/kimera/geoid"} 843
kimera_api_requests_total{endpoint="/kimera/scar"} 621

# HELP kimera_request_duration_seconds Request duration in seconds
# TYPE kimera_request_duration_seconds histogram
kimera_request_duration_seconds_bucket{endpoint="/kimera/status",le="0.1"} 1423
kimera_request_duration_seconds_bucket{endpoint="/kimera/status",le="0.5"} 1520
kimera_request_duration_seconds_bucket{endpoint="/kimera/status",le="1.0"} 1530
kimera_request_duration_seconds_bucket{endpoint="/kimera/status",le="+Inf"} 1532
```

## Error Responses

### Standard Error Format

```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2023-06-14T12:34:56.789Z",
  "path": "/kimera/endpoint",
  "trace_id": "trace-123456"
}
```

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| INVALID_INPUT | Invalid input parameters |
| RESOURCE_NOT_FOUND | Requested resource not found |
| DATABASE_ERROR | Database operation failed |
| AUTHENTICATION_ERROR | Authentication failed |
| AUTHORIZATION_ERROR | Insufficient permissions |
| INTERNAL_ERROR | Internal server error |

## Rate Limiting

The API implements rate limiting to prevent abuse. Rate limits are applied per IP address and authenticated user.

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623670496
```

## Versioning

The API version is included in the response headers:

```
X-Kimera-Version: 0.1.0
```

## Data Types

### Geoid

A Geoid represents a cognitive state as a high-dimensional vector.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Unique identifier |
| timestamp | DateTime | Creation timestamp |
| state_vector | Array | High-dimensional vector representation |
| metadata | Object | Additional metadata |
| entropy | Float | Information entropy of the state |
| coherence_factor | Float | Measure of internal coherence |
| energy_level | Float | Energy level of the state |
| creation_context | String | Context in which the state was created |
| tags | Array | Tags for categorization |

### SCAR (State Change Analysis Record)

A SCAR represents a transition between cognitive states.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Unique identifier |
| source_id | UUID | Source geoid ID |
| target_id | UUID | Target geoid ID |
| transition_energy | Float | Energy required for the transition |
| conservation_error | Float | Error in energy conservation |
| transition_type | String | Type of transition |
| timestamp | DateTime | Creation timestamp |
| metadata | Object | Additional metadata |

### Portal Configuration

A Portal Configuration represents the configuration for interdimensional portals.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Unique identifier |
| source_dimension | Integer | Source dimension |
| target_dimension | Integer | Target dimension |
| radius | Float | Portal radius |
| energy_requirement | Float | Energy required to maintain the portal |
| stability_factor | Float | Stability factor of the portal |
| creation_timestamp | DateTime | Creation timestamp |
| last_used_timestamp | DateTime | Last used timestamp |
| configuration_parameters | Object | Additional configuration parameters |
| status | String | Portal status | 