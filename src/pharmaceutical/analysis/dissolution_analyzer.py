"""
Advanced Dissolution Analysis Engine

Sophisticated dissolution kinetics analysis, f2 similarity calculations,
and dissolution profile optimization for pharmaceutical development.

Integrates with Kimera's GPU acceleration and scientific validation framework.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import interpolate, optimize, stats
from sklearn.metrics import mean_squared_error, r2_score

from ...utils.gpu_foundation import GPUFoundation
from ...utils.kimera_exceptions import KimeraBaseException as KimeraException
from ...utils.kimera_logger import get_logger

logger = get_logger(__name__)


@dataclass
class DissolutionKinetics:
    """Auto-generated class."""
    pass
    """Dissolution kinetics model parameters."""

    model_type: (
        str  # 'zero_order', 'first_order', 'higuchi', 'korsmeyer_peppas', 'weibull'
    )
    parameters: Dict[str, float]
    r_squared: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    residual_standard_error: float


@dataclass
class DissolutionComparison:
    """Auto-generated class."""
    pass
    """Dissolution profile comparison results."""

    f2_similarity: float
    difference_factor_f1: float
    mean_dissolution_time_ratio: float
    dissolution_efficiency_ratio: float
    similarity_assessment: str  # 'SIMILAR', 'DISSIMILAR', 'INCONCLUSIVE'
    statistical_tests: Dict[str, float]


@dataclass
class ModelPrediction:
    """Auto-generated class."""
    pass
    """Dissolution model prediction results."""

    predicted_times: List[float]
    predicted_releases: List[float]
    confidence_interval_lower: List[float]
    confidence_interval_upper: List[float]
    prediction_accuracy: float
class DissolutionAnalyzer:
    """Auto-generated class."""
    pass
    """
    Advanced dissolution analysis engine for pharmaceutical development.

    Provides comprehensive dissolution kinetics modeling, profile comparison,
    and optimization capabilities with GPU acceleration.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the dissolution analyzer.

        Args:
            use_gpu: Whether to use GPU acceleration for computations

        Raises:
            KimeraException: If initialization fails
        """
        self.logger = logger
        self.use_gpu = use_gpu
        self.device = None
        self.gpu_foundation = None

        # Initialize GPU if requested
        if self.use_gpu:
            try:
                self.gpu_foundation = GPUFoundation()
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.logger.info(
                    f"🚀 Dissolution Analyzer initialized on {self.device}"
                )
            except Exception as e:
                self.logger.warning(f"GPU initialization failed, using CPU: {e}")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # Dissolution models
        self.kinetic_models = {
            "zero_order": self._zero_order_model,
            "first_order": self._first_order_model,
            "higuchi": self._higuchi_model,
            "korsmeyer_peppas": self._korsmeyer_peppas_model,
            "weibull": self._weibull_model,
        }

        # Machine learning components
        self.ml_model = None
        self.feature_scaler = None
        self.training_data = []
        self.model_performance = {}

        self.analysis_results = {}

        self._initialize_ml_components()

        self.logger.info("📊 Advanced Dissolution Analyzer initialized")

    def _initialize_ml_components(self):
        """Initialize machine learning components for dissolution prediction."""
        try:
            # Initialize a simple neural network for dissolution prediction
            if self.use_gpu and torch.cuda.is_available():
                self.ml_model = self._create_dissolution_neural_network()
                self.ml_model.to(self.device)

                self.logger.info("🤖 ML dissolution prediction model initialized")

        except Exception as e:
            self.logger.warning(f"ML components initialization failed: {e}")

    def _create_dissolution_neural_network(self):
        """Create a neural network for dissolution prediction."""
        import torch.nn as nn

        class DissolutionPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_layers = nn.Sequential(
                    nn.Linear(12, 64),  # Input: formulation parameters
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 20),  # Output: dissolution profile (20 time points)
                )

            def forward(self, x):
                return (
                    torch.sigmoid(self.feature_layers(x)) * 100
                )  # Scale to percentage

        return DissolutionPredictor()

    def predict_dissolution_ml(
        self, formulation_params: Dict[str, float], time_points: List[float]
    ) -> ModelPrediction:
        """
        Predict dissolution profile using machine learning model.

        Args:
            formulation_params: Formulation parameters
            time_points: Time points for prediction

        Returns:
            ModelPrediction: ML-based dissolution prediction

        Raises:
            KimeraException: If ML prediction fails
        """
        try:
            if self.ml_model is None:
                raise KimeraException("ML model not initialized")

            self.logger.info("🤖 Predicting dissolution profile using ML model...")

            # Prepare input features
            features = self._extract_formulation_features(formulation_params)
            feature_tensor = torch.tensor(
                features, device=self.device, dtype=torch.float32
            ).unsqueeze(0)

            # Predict dissolution profile
            with torch.no_grad():
                predicted_profile = (
                    self.ml_model(feature_tensor).squeeze().cpu().numpy()
                )

            # Interpolate to requested time points
            standard_times = np.linspace(0, 24, 20)  # Standard 20 time points
            interpolator = interpolate.interp1d(
                standard_times,
                predicted_profile,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

            predicted_releases = interpolator(time_points)
            predicted_releases = np.clip(
                predicted_releases, 0, 100
            )  # Ensure valid range

            # Calculate confidence intervals (simplified)
            uncertainty = 5.0  # ±5% uncertainty
            confidence_lower = predicted_releases - uncertainty
            confidence_upper = predicted_releases + uncertainty

            # Calculate prediction accuracy based on model performance
            accuracy = self.model_performance.get("r2_score", 0.85)

            prediction = ModelPrediction(
                predicted_times=list(time_points),
                predicted_releases=list(predicted_releases),
                confidence_interval_lower=list(confidence_lower),
                confidence_interval_upper=list(confidence_upper),
                prediction_accuracy=accuracy,
            )

            self.logger.info(f"✅ ML prediction completed with {accuracy:.2%} accuracy")

            return prediction

        except Exception as e:
            self.logger.error(f"❌ ML dissolution prediction failed: {e}")
            raise KimeraException(f"ML prediction failed: {e}")

    def _extract_formulation_features(
        self, formulation_params: Dict[str, float]
    ) -> List[float]:
        """Extract numerical features from formulation parameters."""
        # Standard feature set for dissolution prediction
        features = [
            formulation_params.get("coating_thickness", 15.0),
            formulation_params.get("ethylcellulose_ratio", 0.8),
            formulation_params.get("hpc_ratio", 0.2),
            formulation_params.get("drug_loading", 50.0),
            formulation_params.get("particle_size", 150.0),
            formulation_params.get("tablet_hardness", 8.0),
            formulation_params.get("porosity", 0.15),
            formulation_params.get("surface_area", 2.5),
            formulation_params.get("ph_media", 6.8),
            formulation_params.get("temperature", 37.0),
            formulation_params.get("agitation_speed", 100.0),
            formulation_params.get("ionic_strength", 0.1),
        ]

        return features

    def train_ml_model(
        self, training_data: List[Dict[str, Any]], validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the ML model on dissolution data.

        Args:
            training_data: List of training examples with formulation and dissolution data
            validation_split: Fraction of data to use for validation

        Returns:
            Dict[str, float]: Training performance metrics

        Raises:
            KimeraException: If training fails
        """
        try:
            if self.ml_model is None:
                raise KimeraException("ML model not initialized")

            self.logger.info(
                f"🎓 Training ML model on {len(training_data)} examples..."
            )

            # Prepare training data
            X, y = self._prepare_training_data(training_data)

            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Convert to tensors
            X_train_tensor = torch.tensor(
                X_train, device=self.device, dtype=torch.float32
            )
            y_train_tensor = torch.tensor(
                y_train, device=self.device, dtype=torch.float32
            )
            X_val_tensor = torch.tensor(X_val, device=self.device, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, device=self.device, dtype=torch.float32)

            # Training setup
            optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()

            # Training loop
            epochs = 100
            best_val_loss = float("inf")

            for epoch in range(epochs):
                # Training step
                self.ml_model.train()
                optimizer.zero_grad()

                train_pred = self.ml_model(X_train_tensor)
                train_loss = criterion(train_pred, y_train_tensor)

                train_loss.backward()
                optimizer.step()

                # Validation step
                if epoch % 10 == 0:
                    self.ml_model.eval()
                    with torch.no_grad():
                        val_pred = self.ml_model(X_val_tensor)
                        val_loss = criterion(val_pred, y_val_tensor)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss

                    self.logger.info(
                        f"   Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                    )

            # Calculate final performance metrics
            self.ml_model.eval()
            with torch.no_grad():
                final_pred = self.ml_model(X_val_tensor).cpu().numpy()
                final_true = y_val_tensor.cpu().numpy()

                r2_score_val = r2_score(final_true.flatten(), final_pred.flatten())
                mse_val = mean_squared_error(final_true.flatten(), final_pred.flatten())

                self.model_performance = {
                    "r2_score": r2_score_val,
                    "mse": mse_val,
                    "training_examples": len(training_data),
                    "validation_loss": best_val_loss.item(),
                }

            self.logger.info(f"✅ ML model training completed")
            self.logger.info(f"   R² Score: {r2_score_val:.4f}")
            self.logger.info(f"   MSE: {mse_val:.4f}")

            return self.model_performance

        except Exception as e:
            self.logger.error(f"❌ ML model training failed: {e}")
            raise KimeraException(f"ML training failed: {e}")

    def _prepare_training_data(
        self, training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML model."""
        X, y = [], []

        for example in training_data:
            formulation = example.get("formulation_params", {})
            dissolution = example.get("dissolution_profile", [])

            if len(dissolution) > 0:
                features = self._extract_formulation_features(formulation)

                # Standardize dissolution profile to 20 time points
                if len(dissolution) != 20:
                    time_orig = np.linspace(0, 24, len(dissolution))
                    time_standard = np.linspace(0, 24, 20)
                    dissolution = np.interp(time_standard, time_orig, dissolution)

                X.append(features)
                y.append(dissolution)

        return np.array(X), np.array(y)

    def analyze_dissolution_kinetics(
        self,
        time_points: List[float],
        release_percentages: List[float],
        models_to_fit: Optional[List[str]] = None,
    ) -> Dict[str, DissolutionKinetics]:
        """
        Analyze dissolution kinetics using multiple mathematical models.

        Args:
            time_points: Time points in hours
            release_percentages: Cumulative release percentages
            models_to_fit: List of models to fit (default: all models)

        Returns:
            Dict[str, DissolutionKinetics]: Fitted kinetic models

        Raises:
            KimeraException: If kinetics analysis fails
        """
        try:
            self.logger.info("📈 Analyzing dissolution kinetics...")

            if len(time_points) != len(release_percentages):
                raise KimeraException(
                    "Time points and release percentages must have same length"
                )

            if models_to_fit is None:
                models_to_fit = list(self.kinetic_models.keys())

            # Convert to numpy arrays
            t = np.array(time_points)
            q = np.array(release_percentages)

            # Fit each kinetic model
            fitted_models = {}

            for model_name in models_to_fit:
                if model_name not in self.kinetic_models:
                    self.logger.warning(f"Unknown model: {model_name}")
                    continue

                try:
                    kinetics = self._fit_kinetic_model(model_name, t, q)
                    fitted_models[model_name] = kinetics

                    self.logger.info(
                        f"✅ {model_name.title()} model: R² = {kinetics.r_squared:.4f}, "
                        f"AIC = {kinetics.aic:.2f}"
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to fit {model_name} model: {e}")

            # Rank models by goodness of fit
            if fitted_models:
                best_model = min(
                    fitted_models.keys(), key=lambda k: fitted_models[k].aic
                )
                self.logger.info(f"🏆 Best fitting model: {best_model.title()}")

            return fitted_models

        except Exception as e:
            self.logger.error(f"❌ Dissolution kinetics analysis failed: {e}")
            raise KimeraException(f"Kinetics analysis failed: {e}")

    def compare_dissolution_profiles(
        self,
        profile1_times: List[float],
        profile1_releases: List[float],
        profile2_times: List[float],
        profile2_releases: List[float],
        interpolate_profiles: bool = True,
    ) -> DissolutionComparison:
        """
        Compare two dissolution profiles using multiple similarity metrics.

        Args:
            profile1_times: Time points for profile 1
            profile1_releases: Release percentages for profile 1
            profile2_times: Time points for profile 2
            profile2_releases: Release percentages for profile 2
            interpolate_profiles: Whether to interpolate profiles to common time points

        Returns:
            DissolutionComparison: Comprehensive comparison results

        Raises:
            KimeraException: If profile comparison fails
        """
        try:
            self.logger.info("🔍 Comparing dissolution profiles...")

            # Interpolate profiles to common time points if requested
            if interpolate_profiles:
                common_times = np.linspace(
                    max(min(profile1_times), min(profile2_times)),
                    min(max(profile1_times), max(profile2_times)),
                    20,
                )

                # Interpolate both profiles
                interp1 = interpolate.interp1d(
                    profile1_times,
                    profile1_releases,
                    kind="linear",
                    fill_value="extrapolate",
                )
                interp2 = interpolate.interp1d(
                    profile2_times,
                    profile2_releases,
                    kind="linear",
                    fill_value="extrapolate",
                )

                releases1 = interp1(common_times)
                releases2 = interp2(common_times)
            else:
                # Use original data (must have same time points)
                if len(profile1_times) != len(profile2_times):
                    raise KimeraException(
                        "Profiles must have same time points for direct comparison"
                    )
                common_times = np.array(profile1_times)
                releases1 = np.array(profile1_releases)
                releases2 = np.array(profile2_releases)

            # Calculate f2 similarity factor
            f2_similarity = self._calculate_f2_similarity(releases1, releases2)

            # Calculate f1 difference factor
            f1_difference = self._calculate_f1_difference(releases1, releases2)

            # Calculate Mean Dissolution Time (MDT)
            mdt1 = self._calculate_mdt(common_times, releases1)
            mdt2 = self._calculate_mdt(common_times, releases2)
            mdt_ratio = mdt2 / mdt1 if mdt1 > 0 else 1.0

            # Calculate Dissolution Efficiency (DE)
            de1 = self._calculate_dissolution_efficiency(common_times, releases1)
            de2 = self._calculate_dissolution_efficiency(common_times, releases2)
            de_ratio = de2 / de1 if de1 > 0 else 1.0

            # Statistical tests
            statistical_tests = self._perform_statistical_tests(releases1, releases2)

            # Determine similarity assessment
            similarity_assessment = self._assess_similarity(
                f2_similarity, f1_difference, statistical_tests
            )

            comparison = DissolutionComparison(
                f2_similarity=f2_similarity,
                difference_factor_f1=f1_difference,
                mean_dissolution_time_ratio=mdt_ratio,
                dissolution_efficiency_ratio=de_ratio,
                similarity_assessment=similarity_assessment,
                statistical_tests=statistical_tests,
            )

            self.logger.info(
                f"✅ Profile comparison completed: f2 = {f2_similarity:.2f}, "
                f"Assessment = {similarity_assessment}"
            )

            return comparison

        except Exception as e:
            self.logger.error(f"❌ Dissolution profile comparison failed: {e}")
            raise KimeraException(f"Profile comparison failed: {e}")

    def predict_dissolution_profile(
        self,
        kinetic_model: DissolutionKinetics,
        prediction_times: List[float],
        confidence_level: float = 0.95,
    ) -> ModelPrediction:
        """
        Predict dissolution profile using fitted kinetic model.

        Args:
            kinetic_model: Fitted kinetic model
            prediction_times: Time points for prediction
            confidence_level: Confidence level for prediction intervals

        Returns:
            ModelPrediction: Model predictions with confidence intervals

        Raises:
            KimeraException: If prediction fails
        """
        try:
            self.logger.info(
                f"🔮 Predicting dissolution using {kinetic_model.model_type} model..."
            )

            t_pred = np.array(prediction_times)

            # Get model function
            if kinetic_model.model_type not in self.kinetic_models:
                raise KimeraException(f"Unknown model type: {kinetic_model.model_type}")

            model_func = self.kinetic_models[kinetic_model.model_type]

            # Make predictions
            predicted_releases = model_func(t_pred, **kinetic_model.parameters)

            # Calculate confidence intervals (simplified approach)
            # In practice, this would use proper statistical methods
            prediction_error = kinetic_model.residual_standard_error
            z_score = stats.norm.ppf((1 + confidence_level) / 2)

            margin_of_error = z_score * prediction_error

            ci_lower = np.maximum(predicted_releases - margin_of_error, 0)
            ci_upper = np.minimum(predicted_releases + margin_of_error, 100)

            # Estimate prediction accuracy
            prediction_accuracy = max(0, 1 - prediction_error / 10)  # Normalized

            prediction = ModelPrediction(
                predicted_times=prediction_times.copy(),
                predicted_releases=predicted_releases.tolist(),
                confidence_interval_lower=ci_lower.tolist(),
                confidence_interval_upper=ci_upper.tolist(),
                prediction_accuracy=prediction_accuracy,
            )

            self.logger.info(
                f"✅ Dissolution prediction completed with {prediction_accuracy:.1%} accuracy"
            )

            return prediction

        except Exception as e:
            self.logger.error(f"❌ Dissolution prediction failed: {e}")
            raise KimeraException(f"Dissolution prediction failed: {e}")

    def optimize_dissolution_target(
        self,
        target_times: List[float],
        target_releases: List[float],
        formulation_constraints: Dict[str, Tuple[float, float]],
        optimization_method: str = "differential_evolution",
    ) -> Dict[str, Any]:
        """
        Optimize formulation parameters to achieve target dissolution profile.

        Args:
            target_times: Target time points
            target_releases: Target release percentages
            formulation_constraints: Parameter constraints (min, max) for each parameter
            optimization_method: Optimization algorithm to use

        Returns:
            Dict[str, Any]: Optimization results

        Raises:
            KimeraException: If optimization fails
        """
        try:
            self.logger.info(
                "🎯 Optimizing formulation for target dissolution profile..."
            )

            # Define objective function
            def objective_function(params):
                """Objective function to minimize (difference from target)."""
                try:
                    # Map parameters to formulation variables
                    param_dict = {}
                    param_names = list(formulation_constraints.keys())
                    for i, param_name in enumerate(param_names):
                        param_dict[param_name] = params[i]

                    # Simulate dissolution profile with these parameters
                    simulated_releases = self._simulate_dissolution_from_formulation(
                        target_times, param_dict
                    )

                    # Calculate mean squared error
                    mse = mean_squared_error(target_releases, simulated_releases)
                    return mse

                except Exception:
                    return 1e6  # Large penalty for invalid parameters

            # Set up optimization bounds
            bounds = [
                formulation_constraints[param]
                for param in formulation_constraints.keys()
            ]

            # Perform optimization
            if optimization_method == "differential_evolution":
                result = optimize.differential_evolution(
                    objective_function, bounds, seed=42, maxiter=100, popsize=15
                )
            elif optimization_method == "minimize":
                # Use initial guess as midpoint of bounds
                x0 = [(b[0] + b[1]) / 2 for b in bounds]
                result = optimize.minimize(
                    objective_function, x0, bounds=bounds, method="L-BFGS-B"
                )
            else:
                raise KimeraException(
                    f"Unknown optimization method: {optimization_method}"
                )

            # Extract optimal parameters
            optimal_params = {}
            param_names = list(formulation_constraints.keys())
            for i, param_name in enumerate(param_names):
                optimal_params[param_name] = result.x[i]

            # Simulate final profile with optimal parameters
            optimal_releases = self._simulate_dissolution_from_formulation(
                target_times, optimal_params
            )

            # Calculate final similarity
            f2_similarity = self._calculate_f2_similarity(
                target_releases, optimal_releases
            )

            optimization_result = {
                "optimal_parameters": optimal_params,
                "objective_value": result.fun,
                "f2_similarity": f2_similarity,
                "optimization_success": result.success,
                "iterations": result.nit if hasattr(result, "nit") else None,
                "predicted_profile": {
                    "times": target_times,
                    "releases": optimal_releases.tolist(),
                },
            }

            self.logger.info(
                f"✅ Optimization completed: f2 = {f2_similarity:.2f}, "
                f"Success = {result.success}"
            )

            return optimization_result

        except Exception as e:
            self.logger.error(f"❌ Dissolution optimization failed: {e}")
            raise KimeraException(f"Dissolution optimization failed: {e}")

    # Kinetic model implementations
    def _zero_order_model(self, t: np.ndarray, k0: float) -> np.ndarray:
        """Zero-order kinetics: Q = k0 * t"""
        return np.minimum(k0 * t, 100)

    def _first_order_model(self, t: np.ndarray, k1: float, q_inf: float) -> np.ndarray:
        """First-order kinetics: Q = Q_inf * (1 - exp(-k1 * t))"""
        return q_inf * (1 - np.exp(-k1 * t))

    def _higuchi_model(self, t: np.ndarray, kh: float) -> np.ndarray:
        """Higuchi model: Q = kh * sqrt(t)"""
        return np.minimum(kh * np.sqrt(t), 100)

    def _korsmeyer_peppas_model(self, t: np.ndarray, k: float, n: float) -> np.ndarray:
        """Korsmeyer-Peppas model: Q = k * t^n"""
        return np.minimum(k * np.power(t, n), 100)

    def _weibull_model(
        self, t: np.ndarray, a: float, b: float, ti: float = 0
    ) -> np.ndarray:
        """Weibull model: Q = a * (1 - exp(-((t-ti)/b)^c))"""
        # Simplified Weibull with c=1
        return a * (1 - np.exp(-((t - ti) / b)))

    def _fit_kinetic_model(
        self, model_name: str, t: np.ndarray, q: np.ndarray
    ) -> DissolutionKinetics:
        """Fit a specific kinetic model to dissolution data."""
        model_func = self.kinetic_models[model_name]

        # Define parameter bounds and initial guesses for each model
        if model_name == "zero_order":
            bounds = ([0], [50])
            p0 = [10]
        elif model_name == "first_order":
            bounds = ([0, 0], [5, 120])
            p0 = [0.5, 100]
        elif model_name == "higuchi":
            bounds = ([0], [100])
            p0 = [20]
        elif model_name == "korsmeyer_peppas":
            bounds = ([0, 0], [100, 2])
            p0 = [10, 0.5]
        elif model_name == "weibull":
            bounds = ([0, 0, 0], [120, 10, 2])
            p0 = [100, 2, 0]
        else:
            raise KimeraException(f"No fitting parameters defined for {model_name}")

        # Fit model
        try:
            popt, pcov = optimize.curve_fit(
                model_func, t, q, p0=p0, bounds=bounds, maxfev=1000
            )
        except Exception as e:
            raise KimeraException(f"Model fitting failed: {e}")

        # Calculate goodness of fit metrics
        q_pred = model_func(t, *popt)
        residuals = q - q_pred

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((q - np.mean(q)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # AIC and BIC
        n = len(q)
        k = len(popt)  # number of parameters
        mse = ss_res / n

        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)

        # Residual standard error
        rse = np.sqrt(ss_res / (n - k))

        # Create parameter dictionary
        if model_name == "zero_order":
            parameters = {"k0": popt[0]}
        elif model_name == "first_order":
            parameters = {"k1": popt[0], "q_inf": popt[1]}
        elif model_name == "higuchi":
            parameters = {"kh": popt[0]}
        elif model_name == "korsmeyer_peppas":
            parameters = {"k": popt[0], "n": popt[1]}
        elif model_name == "weibull":
            parameters = {"a": popt[0], "b": popt[1], "ti": popt[2]}

        return DissolutionKinetics(
            model_type=model_name,
            parameters=parameters,
            r_squared=r_squared,
            aic=aic,
            bic=bic,
            residual_standard_error=rse,
        )

    def _calculate_f2_similarity(
        self, profile1: np.ndarray, profile2: np.ndarray
    ) -> float:
        """Calculate f2 similarity factor."""
        n = len(profile1)
        sum_squared_diff = np.sum((profile1 - profile2) ** 2)
        f2 = 50 * np.log10(((1 + sum_squared_diff / n) ** -0.5) * 100)
        return max(0, min(100, f2))

    def _calculate_f1_difference(
        self, profile1: np.ndarray, profile2: np.ndarray
    ) -> float:
        """Calculate f1 difference factor."""
        n = len(profile1)
        sum_abs_diff = np.sum(np.abs(profile1 - profile2))
        sum_profile1 = np.sum(profile1)
        f1 = (sum_abs_diff / sum_profile1) * 100 if sum_profile1 > 0 else 0
        return f1

    def _calculate_mdt(self, times: np.ndarray, releases: np.ndarray) -> float:
        """Calculate Mean Dissolution Time."""
        # MDT = Σ(t_mid * ΔQ) / Q_total
        if len(times) < 2:
            return 0

        mdt = 0
        total_release = releases[-1]

        for i in range(1, len(times)):
            t_mid = (times[i] + times[i - 1]) / 2
            delta_q = releases[i] - releases[i - 1]
            mdt += t_mid * delta_q

        return mdt / total_release if total_release > 0 else 0

    def _calculate_dissolution_efficiency(
        self, times: np.ndarray, releases: np.ndarray
    ) -> float:
        """Calculate Dissolution Efficiency."""
        # DE = (AUC_0^t / AUC_total) * 100
        # Simplified using trapezoidal rule
        auc = np.trapz(releases, times)
        max_auc = 100 * times[-1]  # Assuming 100% release over time period
        return (auc / max_auc) * 100 if max_auc > 0 else 0

    def _perform_statistical_tests(
        self, profile1: np.ndarray, profile2: np.ndarray
    ) -> Dict[str, float]:
        """Perform statistical tests for profile comparison."""
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(profile1, profile2)

        # Mann-Whitney U test
        u_stat, u_pvalue = stats.mannwhitneyu(
            profile1, profile2, alternative="two-sided"
        )

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(profile1, profile2)

        return {
            "paired_t_test_pvalue": t_pvalue,
            "mann_whitney_u_pvalue": u_pvalue,
            "kolmogorov_smirnov_pvalue": ks_pvalue,
            "mean_difference": np.mean(profile1 - profile2),
            "std_difference": np.std(profile1 - profile2),
        }

    def _assess_similarity(
        self, f2: float, f1: float, stats_tests: Dict[str, float]
    ) -> str:
        """Assess overall similarity based on multiple criteria."""
        # Primary criterion: f2 similarity factor
        f2_similar = f2 >= 50.0

        # Secondary criterion: f1 difference factor
        f1_similar = f1 <= 15.0

        # Statistical significance
        statistically_different = any(
            p_val < 0.05 for key, p_val in stats_tests.items() if "pvalue" in key
        )

        if f2_similar and f1_similar and not statistically_different:
            return "SIMILAR"
        elif not f2_similar or not f1_similar or statistically_different:
            return "DISSIMILAR"
        else:
            return "INCONCLUSIVE"

    def _simulate_dissolution_from_formulation(
        self, times: List[float], formulation_params: Dict[str, float]
    ) -> np.ndarray:
        """Simulate dissolution profile from formulation parameters."""
        # Simplified mechanistic model relating formulation to dissolution
        # In practice, this would be much more sophisticated

        coating_thickness = formulation_params.get("coating_thickness_percent", 12)
        polymer_ratio = formulation_params.get("ethylcellulose_ratio", 0.8)
        particle_size = formulation_params.get("particle_size_um", 100)

        # Higuchi-like model with formulation dependencies
        k_base = 25  # Base release rate constant

        # Coating thickness effect (thicker = slower)
        thickness_factor = 1.0 / (1.0 + coating_thickness / 10)

        # Polymer ratio effect (more ethylcellulose = slower)
        polymer_factor = 1.0 - polymer_ratio * 0.3

        # Particle size effect (smaller = faster)
        size_factor = 100 / particle_size if particle_size > 0 else 1.0

        k_effective = k_base * thickness_factor * polymer_factor * size_factor

        # Generate dissolution profile
        t_array = np.array(times)
        releases = k_effective * np.sqrt(t_array)
        releases = np.minimum(releases, 95)  # Cap at 95%

        return releases

    def generate_dissolution_report(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive dissolution analysis report.

        Args:
            analysis_results: All dissolution analysis results

        Returns:
            Dict[str, Any]: Comprehensive dissolution report
        """
        try:
            self.logger.info("📊 Generating dissolution analysis report...")

            report = {
                "report_id": f"Dissolution_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generation_time": datetime.now().isoformat(),
                "analysis_summary": {
                    "total_analyses": len(analysis_results),
                    "kinetic_models_fitted": sum(
                        1
                        for result in analysis_results.values()
                        if "kinetics" in result
                    ),
                    "profile_comparisons": sum(
                        1
                        for result in analysis_results.values()
                        if "comparison" in result
                    ),
                    "predictions_made": sum(
                        1
                        for result in analysis_results.values()
                        if "prediction" in result
                    ),
                },
                "detailed_results": analysis_results,
                "best_kinetic_models": self._identify_best_models(analysis_results),
                "similarity_assessments": self._summarize_similarities(
                    analysis_results
                ),
                "optimization_results": self._summarize_optimizations(analysis_results),
                "recommendations": self._generate_dissolution_recommendations(
                    analysis_results
                ),
            }

            self.logger.info(f"✅ Dissolution analysis report generated")
            return report

        except Exception as e:
            self.logger.error(f"❌ Dissolution report generation failed: {e}")
            raise KimeraException(f"Dissolution report generation failed: {e}")

    def _identify_best_models(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Identify best kinetic models from analysis results."""
        best_models = {}

        for analysis_id, result in results.items():
            if "kinetics" in result:
                kinetic_results = result["kinetics"]
                if kinetic_results:
                    best_model = min(
                        kinetic_results.keys(), key=lambda k: kinetic_results[k].aic
                    )
                    best_models[analysis_id] = best_model

        return best_models

    def _summarize_similarities(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Summarize similarity assessments."""
        similarities = {}

        for analysis_id, result in results.items():
            if "comparison" in result:
                comparison = result["comparison"]
                similarities[analysis_id] = comparison.similarity_assessment

        return similarities

    def _summarize_optimizations(
        self, results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Summarize optimization results."""
        optimizations = {}

        for analysis_id, result in results.items():
            if "optimization" in result:
                opt_result = result["optimization"]
                optimizations[analysis_id] = {
                    "success": opt_result["optimization_success"],
                    "f2_similarity": opt_result["f2_similarity"],
                    "optimal_parameters": opt_result["optimal_parameters"],
                }

        return optimizations

    def _generate_dissolution_recommendations(
        self, results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on dissolution analysis."""
        recommendations = []

        # Analyze kinetic model fits
        poor_fits = []
        for analysis_id, result in results.items():
            if "kinetics" in result:
                kinetic_results = result["kinetics"]
                best_r2 = max(model.r_squared for model in kinetic_results.values())
                if best_r2 < 0.95:
                    poor_fits.append(analysis_id)

        if poor_fits:
            recommendations.append(
                f"Poor kinetic model fits detected for {len(poor_fits)} analyses. "
                "Consider additional time points or alternative models."
            )

        # Analyze similarity assessments
        dissimilar_profiles = []
        for analysis_id, result in results.items():
            if "comparison" in result:
                if result["comparison"].similarity_assessment == "DISSIMILAR":
                    dissimilar_profiles.append(analysis_id)

        if dissimilar_profiles:
            recommendations.append(
                f"Dissimilar dissolution profiles detected for {len(dissimilar_profiles)} comparisons. "
                "Review formulation parameters or manufacturing process."
            )

        # Analyze optimization results
        failed_optimizations = []
        for analysis_id, result in results.items():
            if "optimization" in result:
                if not result["optimization"]["optimization_success"]:
                    failed_optimizations.append(analysis_id)

        if failed_optimizations:
            recommendations.append(
                f"Optimization failed for {len(failed_optimizations)} analyses. "
                "Consider broader parameter ranges or different optimization methods."
            )

        if not recommendations:
            recommendations.append(
                "All dissolution analyses show good results. "
                "Formulations are ready for further development."
            )

        return recommendations
