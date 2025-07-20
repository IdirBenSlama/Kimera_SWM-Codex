import numpy as np
from typing import List, Dict

class ThermodynamicEngine:
    """
    A scientific engine for calculating thermodynamic properties of cognitive fields.

    This engine provides methods to analyze collections of embeddings (cognitive fields)
    using principles derived from statistical thermodynamics.
    """

    def calculate_semantic_temperature(self, cognitive_field: List[np.ndarray]) -> float:
        """
        Calculates the Semantic Temperature of a cognitive field.

        Temperature is defined as the trace of the covariance matrix of the
        embedding vectors in the field. This measures the semantic dispersion.

        Args:
            cognitive_field: A list of numpy arrays, where each array is an embedding.

        Returns:
            The semantic temperature of the field as a float. Returns 0.0 if the
            field is empty or contains insufficient data for covariance calculation.
        """
        if not cognitive_field or len(cognitive_field) < 2:
            return 0.0

        field_matrix = np.array(cognitive_field)
        
        # Ensure the matrix is 2D
        if field_matrix.ndim == 1:
            field_matrix = field_matrix.reshape(-1, 1)

        cov_matrix = np.cov(field_matrix, rowvar=False)
        
        # For a 1D array of embeddings, np.cov returns a float, not a matrix.
        if cov_matrix.ndim == 0:
            return float(cov_matrix)
            
        temperature = np.trace(cov_matrix)
        return float(temperature)

    def run_semantic_carnot_engine(
        self, hot_reservoir: List[np.ndarray], cold_reservoir: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Runs a theoretical semantic Carnot engine between two cognitive fields.

        This calculates the maximum theoretical efficiency and work extractable
        from the temperature difference between a hot (high entropy) and cold
        (low entropy) cognitive field.

        Args:
            hot_reservoir: A list of embeddings representing the high-temperature source.
            cold_reservoir: A list of embeddings representing the low-temperature sink.

        Returns:
            A dictionary containing the thermodynamic properties of the cycle.
        """
        t_hot = self.calculate_semantic_temperature(hot_reservoir)
        t_cold = self.calculate_semantic_temperature(cold_reservoir)

        if t_hot <= 0 or t_hot <= t_cold:
            efficiency = 0.0
        else:
            efficiency = 1 - (t_cold / t_hot)

        # "Input Heat" (Q_hot) is defined as the total semantic energy of the hot reservoir,
        # which is its temperature.
        q_hot = t_hot
        work_extracted = efficiency * q_hot

        return {
            "carnot_efficiency": efficiency,
            "work_extracted": work_extracted,
            "t_hot": t_hot,
            "t_cold": t_cold,
        } 