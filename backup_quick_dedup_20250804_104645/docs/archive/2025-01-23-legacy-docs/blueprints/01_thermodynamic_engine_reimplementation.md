# Blueprint: Thermodynamic Engine Re-Implementation

**Document ID:** `bp_thermo_01`
**Phase:** 1 (Scientific Core Re-Implementation)
**Status:** In Progress
**Date:** 2024-07-22

---

## 1.0 Objective

This document outlines the plan to re-implement and validate the core thermodynamic analysis capabilities of the Kimera SWM. The goal is to resurrect the system's foundational scientific claim: the ability to perform thermodynamic calculations on cognitive data fields.

This effort is a ground-up rebuild based on the conceptual blueprint reverse-engineered from the non-functional script `archive/broken_scripts_and_tests/scientific_validation/zetetic_real_world_thermodynamic_audit.py`.

## 2.0 Architectural Placement

The new engine will be located at:
*   `backend/engines/thermodynamic_engine.py`

The new validation suite will be located at:
*   `tests/validation/test_thermodynamic_engine.py`

## 3.0 Core Components to Be Implemented

### 3.1 `ThermodynamicEngine` Class (`thermodynamic_engine.py`)

A new class, `ThermodynamicEngine`, will be created to encapsulate all thermodynamic calculations. It will operate on collections of "Geoids" (represented as embedding vectors) retrieved from the `VaultManager`.

#### 3.1.1 `calculate_semantic_temperature`

*   **Signature:** `calculate_semantic_temperature(cognitive_field: List[np.ndarray]) -> float`
*   **Scientific Basis:** In physics, temperature is a measure of the average kinetic energy of particles in a system. For a cognitive field represented by a set of embedding vectors, we will define "Semantic Temperature" as a measure of the semantic dispersion or variance of the concepts within that field.
*   **Formula:** The temperature will be calculated as the **trace of the covariance matrix** of the embedding vectors.
    *   Let \(X\) be a matrix where each row is an embedding vector.
    *   Compute the covariance matrix: \(\Sigma = \text{cov}(X)\)
    *   The temperature is: \(T = \text{Tr}(\Sigma)\)
*   **Justification:** A field of random, unrelated concepts (high entropy) will have high variance in many dimensions, resulting in a high-trace covariance matrix and thus a high temperature. A field of structured, similar concepts (low entropy) will have low variance and a lower temperature. This provides a robust, quantitative, and scientifically grounded metric.

#### 3.1.2 `run_semantic_carnot_engine`

*   **Signature:** `run_semantic_carnot_engine(hot_reservoir: List[np.ndarray], cold_reservoir: List[np.ndarray]) -> Dict[str, float]`
*   **Scientific Basis:** The Carnot cycle is a theoretical thermodynamic cycle of maximum possible efficiency. The efficiency is determined solely by the temperatures of the hot and cold reservoirs.
*   **Implementation Steps:**
    1.  Calculate \(T_{\text{hot}}\) using `calculate_semantic_temperature` on the `hot_reservoir`.
    2.  Calculate \(T_{\text{cold}}\) using `calculate_semantic_temperature` on the `cold_reservoir`.
    3.  Calculate the theoretical maximum **Carnot Efficiency**: 
        \[ \eta_{\text{Carnot}} = 1 - \frac{T_{\text{cold}}}{T_{\text{hot}}} \]
    4.  Define the "input heat" \(Q_{\text{hot}}\) from the hot reservoir as its total semantic energy, which we will define as its temperature \(T_{\text{hot}}\).
    5.  Calculate the theoretical "work" extractable: \(W_{\text{out}} = \eta_{\text{Carnot}} \times Q_{\text{hot}}\).
*   **Return Value:** A dictionary containing `{ "carnot_efficiency": ..., "work_extracted": ..., "t_hot": ..., "t_cold": ... }`.

### 3.2 Validation Suite (`test_thermodynamic_engine.py`)

A new test suite will be created to perform a real-world validation of the `ThermodynamicEngine`. This will be the spiritual successor to the `ZeteticRealWorldThermodynamicAuditor`.

**Test Case: `test_carnot_engine_validation`**

1.  **Setup:**
    *   Instantiate the `ThermodynamicEngine`.
    *   Create a "hot reservoir": A list of 50-100 high-dimensional `np.ndarray` vectors drawn from a random normal distribution (high entropy).
    *   Create a "cold reservoir": A list of 50-100 high-dimensional `np.ndarray` vectors representing a structured concept (e.g., vectors generated from a sine wave, resulting in low entropy).
2.  **Execution:**
    *   Call `run_semantic_carnot_engine` with the prepared reservoirs.
3.  **Validation (Assertions):**
    *   Assert that \(T_{\text{hot}} > T_{\text{cold}}\).
    *   Assert that the calculated `carnot_efficiency` is between 0 and 1.
    *   Assert that the returned values for efficiency, work, and temperatures are floats and not NaN or infinity.
    *   Log all results to the console in a clear, readable format.

## 4.0 Success Criteria for Phase 1

Phase 1 will be considered complete when the `test_carnot_engine_validation` test case runs successfully, its assertions pass, and it produces a verifiable, scientifically plausible result, demonstrating the successful re-implementation of the core thermodynamic claim. 