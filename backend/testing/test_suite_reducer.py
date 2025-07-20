"""
Unsupervised Test Suite Reducer for Kimera
==========================================

This module provides a mechanism for reducing a given test suite using
unsupervised machine learning techniques, specifically TF-IDF vectorization
and K-means clustering. The primary goal is to select a representative
subset of tests that preserves fault-detection capabilities while
significantly reducing execution time.

This implementation is guided by a zetetic, scientific engineering mindset,
starting with a foundational approach (TF-IDF on test names) that can be
iteratively improved with more sophisticated feature extraction methods
(e.g., code coverage analysis).

Core Dependencies:
------------------
- scikit-learn: For machine learning algorithms.
  (pip install scikit-learn)
- numpy: For numerical operations.

Author: Kimera Development Team (AI-Assisted)
Version: 0.1.0 - Initial Implementation
"""

import logging
import os
import json
import random
from typing import List, Dict, Callable, Any, Tuple

# Note: The following imports require scikit-learn and numpy.
# Ensure these are installed in your environment.
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin_min
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None  # Define np as None when numpy is not available

from backend.utils.kimera_logger import get_system_logger
from backend.utils.file_utils import find_project_root, ensure_dir
from backend.config import get_settings

logger = get_system_logger(__name__)


class TestSuiteReducer:
    """
    Reduces a test suite by clustering tests based on their names and
    selecting a representative from each cluster.
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        """
        Initializes the TestSuiteReducer.

        :param n_clusters: The target number of clusters (and thus, tests)
                           for the reduced suite.
        :param random_state: Seed for reproducibility of clustering.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn or numpy is not installed. "
                            "Please install them using 'pip install scikit-learn numpy'")

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z_0-9]+\b')
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        logger.info(f"TestSuiteReducer initialized with n_clusters={n_clusters}")

    def reduce_suite(self, tests: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Reduces the provided test suite.

        :param tests: A dictionary where keys are test names and values are
                      the callable test methods.
        :return: A reduced dictionary of tests.
        """
        test_names = list(tests.keys())
        if len(test_names) <= self.n_clusters:
            logger.warning(
                f"Number of tests ({len(test_names)}) is less than or equal "
                f"to n_clusters ({self.n_clusters}). Skipping reduction."
            )
            return tests

        logger.info(f"Starting test suite reduction from {len(test_names)} tests to {self.n_clusters} clusters.")

        # 1. Feature Engineering: Convert test names to vectors
        try:
            feature_vectors = self._extract_features(test_names)
            logger.debug(f"Successfully created {feature_vectors.shape[0]} feature vectors of dimension {feature_vectors.shape[1]}.")
        except Exception as e:
            logger.error(f"Failed to extract features from test names: {e}")
            # Fallback: return original suite if feature extraction fails
            return tests

        # 2. Unsupervised Clustering
        try:
            self.clusterer.fit(feature_vectors)
            logger.debug("Successfully performed K-means clustering on test feature vectors.")
        except Exception as e:
            logger.error(f"Failed to cluster tests: {e}")
            # Fallback: return original suite if clustering fails
            return tests

        # 3. Representative Selection
        try:
            reduced_test_names = self._select_representatives(feature_vectors, test_names)
            reduced_suite = {name: tests[name] for name in reduced_test_names}
            logger.info(f"Test suite successfully reduced to {len(reduced_suite)} tests.")
            return reduced_suite
        except Exception as e:
            logger.error(f"Failed to select representative tests: {e}")
            # Fallback: return original suite if selection fails
            return tests

    def _extract_features(self, test_names: List[str]) -> np.ndarray:
        """
        Converts a list of test names into a TF-IDF feature matrix.
        We replace underscores with spaces to improve tokenization.
        e.g., 'test_resnet50_inference' -> 'test resnet50 inference'
        """
        processed_names = [name.replace('_', ' ') for name in test_names]
        return self.vectorizer.fit_transform(processed_names)

    def _select_representatives(self, feature_vectors: np.ndarray, test_names: List[str]) -> List[str]:
        """
        Selects one test from each cluster that is closest to the centroid.

        :param feature_vectors: The feature matrix of all tests.
        :param test_names: The list of all test names, in the same order as the feature matrix.
        :return: A list of names for the selected representative tests.
        """
        # Find the index of the closest test to each cluster centroid
        closest, _ = pairwise_distances_argmin_min(self.clusterer.cluster_centers_, feature_vectors)
        
        # Get the names of these tests
        representative_names = [test_names[i] for i in closest]
        
        # Log the clusters and their representatives
        for i, center in enumerate(self.clusterer.cluster_centers_):
            cluster_indices = np.where(self.clusterer.labels_ == i)[0]
            cluster_tests = [test_names[j] for j in cluster_indices]
            logger.debug(f"Cluster {i} Representative: '{test_names[closest[i]]}'")
            logger.debug(f"Cluster {i} Members: {cluster_tests}")
            
        return representative_names 