"""
Kimera Cognitive Cycle Logic (KCCL)
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from ..config.settings import get_settings
from ..core.embedding_utils import encode_text
from ..core.scar import ScarRecord
from ..utils.config import get_api_settings
from ..utils.kimera_logger import get_cognitive_logger

# Initialize logger
logger = get_cognitive_logger(__name__)


@dataclass
class KimeraCognitiveCycle:
    """Minimal cognitive loop used for the test suite."""

    def run_cycle(self, system: dict) -> str:
        """Execute one cognitive cycle over the provided system."""

        try:
            spde = system["spde_engine"]
            contradiction_engine = system["contradiction_engine"]
            vault_manager = system["vault_manager"]

            cycle_stats = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "contradictions_detected": 0,
                "scars_created": 0,
                "entropy_before_diffusion": 0.0,
                "entropy_after_diffusion": 0.0,
                "entropy_delta": 0.0,
                "errors_encountered": 0,
                "geoids_processed": 0,
            }

            # Safety check: limit processing if too many geoids
            active_geoids = system["active_geoids"]
            geoids_to_process = list(active_geoids.values())

            if len(geoids_to_process) > 500:  # Safety limit
                logging.warning(
                    f"Large geoid count ({len(geoids_to_process)}), limiting to 500 for cycle"
                )
                geoids_to_process = geoids_to_process[:500]

            cycle_stats["geoids_processed"] = len(geoids_to_process)

            # --- Semantic Pressure Diffusion ---
            try:
                entropy_before = sum(g.calculate_entropy() for g in geoids_to_process)

                for geoid in geoids_to_process:
                    try:
                        geoid.semantic_state = spde.diffuse(geoid.semantic_state)
                    except Exception as e:
                        cycle_stats["errors_encountered"] += 1
                        # Continue processing other geoids
                        continue

                entropy_after = sum(g.calculate_entropy() for g in geoids_to_process)

                cycle_stats["entropy_before_diffusion"] = entropy_before
                cycle_stats["entropy_after_diffusion"] = entropy_after
                cycle_stats["entropy_delta"] = entropy_after - entropy_before

            except Exception as e:
                logging.error(f"SPDE diffusion failed: {e}")
                cycle_stats["errors_encountered"] += 1

            # --- Contradiction Detection ---
            try:
                tensions = contradiction_engine.detect_tension_gradients(
                    geoids_to_process
                )
                cycle_stats["contradictions_detected"] = len(tensions)

                # Limit tension processing to prevent overload
                tensions_to_process = tensions[:20]  # Process max 20 tensions per cycle

                for tension in tensions_to_process:
                    try:
                        summary = f"Tension {tension.geoid_a}-{tension.geoid_b}"
                        vector = encode_text(summary)
                        scar = ScarRecord(
                            scar_id=f"SCAR_{uuid.uuid4().hex[:8]}",
                            geoids=[tension.geoid_a, tension.geoid_b],
                            reason="auto-cycle",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            resolved_by="KimeraCognitiveCycle",
                            pre_entropy=0.0,
                            post_entropy=0.0,
                            delta_entropy=0.0,
                            cls_angle=tension.tension_score * 180,
                            semantic_polarity=0.0,
                            mutation_frequency=tension.tension_score,
                        )
                        vault_manager.insert_scar(scar, vector)
                        cycle_stats["scars_created"] += 1
                    except Exception as e:
                        logging.warning(
                            f"Failed to process tension {tension.geoid_a}-{tension.geoid_b}: {e}"
                        )
                        cycle_stats["errors_encountered"] += 1
                        continue

            except Exception as e:
                logging.error(f"Contradiction detection failed: {e}")
                cycle_stats["errors_encountered"] += 1

            # --- Cycle bookkeeping ---
            state = system.setdefault("system_state", {})
            state["cycle_count"] = state.get("cycle_count", 0) + 1
            state["last_cycle"] = cycle_stats

            # --- Revolutionary Thermodynamic Integration (Phase 4.0) ---
            try:
                # Run thermodynamic consciousness detection if available
                foundational_engine = system.get("revolutionary_thermodynamics_engine")
                consciousness_detector = system.get("consciousness_detector")

                THERMODYNAMIC_INTERVAL = 3  # Run every 3 cycles for efficiency
                if (
                    foundational_engine
                    and state["cycle_count"] % THERMODYNAMIC_INTERVAL == 0
                ):
                    # Run thermodynamic optimization on processed geoids
                    if geoids_to_process:
                        try:
                            # Detect complexity threshold
                            if consciousness_detector and len(geoids_to_process) >= 2:
                                complexity_state = (
                                    foundational_engine.detect_complexity_threshold(
                                        geoids_to_process[:5]
                                    )
                                )
                                cycle_stats["complexity_detected"] = (
                                    complexity_state.complexity_probability >= 0.7
                                )
                                cycle_stats["complexity_probability"] = (
                                    complexity_state.complexity_probability
                                )
                                cycle_stats["temperature_coherence"] = (
                                    complexity_state.temperature_coherence
                                )

                                if complexity_state.complexity_probability >= 0.7:
                                    logging.info(
                                        f"🔬 High complexity threshold detected! Probability: {complexity_state.complexity_probability:.3f}"
                                    )

                            # Run Zetetic Carnot optimization if sufficient geoids
                            if len(geoids_to_process) >= 4:
                                mid_point = len(geoids_to_process) // 2
                                hot_geoids = geoids_to_process[:mid_point]
                                cold_geoids = geoids_to_process[mid_point:]

                                carnot_result = (
                                    foundational_engine.run_zetetic_carnot_engine(
                                        hot_geoids, cold_geoids
                                    )
                                )
                                cycle_stats["thermodynamic_work_extracted"] = (
                                    carnot_result.work_extracted
                                )
                                cycle_stats["thermodynamic_efficiency"] = (
                                    carnot_result.actual_efficiency
                                )
                                cycle_stats["physics_compliant"] = (
                                    carnot_result.physics_compliant
                                )

                                if not carnot_result.physics_compliant:
                                    logging.warning(
                                        f"⚠️ Physics violation detected in cycle {state['cycle_count']}"
                                    )

                        except Exception as thermo_exc:
                            logging.warning(
                                f"Thermodynamic processing failed: {thermo_exc}"
                            )
                            cycle_stats["thermodynamic_errors"] = (
                                cycle_stats.get("thermodynamic_errors", 0) + 1
                            )

            except Exception as exc:
                # Log but never crash the cognitive cycle
                logging.getLogger(__name__).warning(
                    "Revolutionary Thermodynamic Integration failed: %s",
                    exc,
                    exc_info=True,
                )
                cycle_stats["errors_encountered"] += 1

            # --- Meta-Insight Hook (Phase 3.5) ---
            try:
                meta_engine = system.get("meta_insight_engine")
                recent_insights = system.get("recent_insights", [])

                META_INTERVAL = 5  # trigger every N cycles
                if meta_engine and state["cycle_count"] % META_INTERVAL == 0:
                    meta_insights = meta_engine.scan_recent_insights(recent_insights)
                    if meta_insights:
                        # Append generated meta-insights to recent insights list
                        recent_insights.extend(meta_insights)
                        # Keep list size reasonable (latest 100)
                        if len(recent_insights) > 100:
                            del recent_insights[: len(recent_insights) - 100]
            except Exception as exc:
                # Log but never crash the cognitive cycle
                logging.getLogger(__name__).warning(
                    "Meta-Insight Engine failed: %s", exc, exc_info=True
                )
                cycle_stats["errors_encountered"] += 1

            # Determine cycle status based on errors
            if cycle_stats["errors_encountered"] == 0:
                return "cycle complete"
            elif (
                cycle_stats["errors_encountered"]
                < cycle_stats["geoids_processed"] * 0.1
            ):  # Less than 10% error rate
                return "cycle partial"
            else:
                return "cycle degraded"

        except Exception as e:
            # Catastrophic failure - log and return error status
            logger.error(f"Cognitive cycle catastrophic failure: {e}")

            # Ensure basic cycle bookkeeping even on failure
            state = system.setdefault("system_state", {})
            state["cycle_count"] = state.get("cycle_count", 0) + 1
            state["last_cycle"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "contradictions_detected": 0,
                "scars_created": 0,
                "entropy_before_diffusion": 0.0,
                "entropy_after_diffusion": 0.0,
                "entropy_delta": 0.0,
                "errors_encountered": 1,
                "geoids_processed": 0,
                "failure_reason": str(e),
            }

            return "cycle failed"
