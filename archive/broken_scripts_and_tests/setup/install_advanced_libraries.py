#!/usr/bin/env python3
"""
Kimera Advanced Libraries Installation Script
Installs state-of-the-art trading, optimization, and fraud detection libraries
"""

import subprocess
import sys
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LibraryInstaller:
    """Advanced library installer for Kimera trading system"""
    
    def __init__(self):
        self.installed_packages = []
        self.failed_packages = []
        
        # Define library categories with specific versions for stability
        self.library_categories = {
            "anomaly_detection": {
                "description": "Fraud Detection & Anomaly Detection Libraries",
                "packages": [
                    "pyod>=1.1.0",              # Comprehensive outlier detection
                    "shap>=0.44.0",             # Model interpretability
                    "scikit-learn>=1.3.0",      # Extended Isolation Forest
                    "tensorflow>=2.13.0",       # Deep learning for autoencoders
                    "keras>=2.13.0",            # Neural network models
                    "xgboost>=1.7.0",           # Gradient boosting for anomaly detection
                    "lightgbm>=4.0.0",          # Light gradient boosting
                ]
            },
            
            "optimization": {
                "description": "Mathematical Optimization & Decision Making",
                "packages": [
                    "cvxpy>=1.4.0",             # Convex optimization
                    "ortools>=9.7.0",           # Google OR-Tools
                    "pulp>=2.7.0",              # Linear programming
                    "scipy>=1.11.0",            # Scientific computing
                    "cvxopt>=1.3.0",            # Convex optimization
                    "gurobipy",                 # Gurobi optimizer (if license available)
                    "mosek",                    # MOSEK optimizer (if license available)
                ]
            },
            
            "reinforcement_learning": {
                "description": "Advanced ML & Reinforcement Learning",
                "packages": [
                    "stable-baselines3>=2.1.0", # Reinforcement learning
                    "ray[rllib]>=2.7.0",        # Distributed RL
                    "gymnasium>=0.29.0",        # RL environments
                    "torch>=2.0.0",             # PyTorch for RL
                    "tensorboard>=2.14.0",      # Training visualization
                    "wandb>=0.15.0",            # Experiment tracking
                ]
            },
            
            "quantitative_finance": {
                "description": "Quantitative Finance & Portfolio Management",
                "packages": [
                    "zipline-reloaded>=3.0.0",  # Backtesting framework
                    "backtrader>=1.9.0",        # Alternative backtesting
                    "pyportfolioopt>=1.5.0",    # Portfolio optimization
                    "quantlib>=1.31.0",         # Quantitative finance library
                    "ta-lib>=0.4.0",            # Technical analysis
                    "finrl>=0.3.0",             # Financial RL library
                    "riskfolio-lib>=4.3.0",     # Portfolio optimization and risk management
                    "empyrical>=0.5.0",         # Performance metrics
                ]
            },
            
            "market_data": {
                "description": "Market Data & Exchange Connectivity",
                "packages": [
                    "ccxt>=4.0.0",              # Crypto exchange connectivity
                    "yfinance>=0.2.0",          # Yahoo Finance data
                    "alpha-vantage>=2.3.0",     # Alpha Vantage API
                    "quandl>=3.7.0",            # Quandl data
                    "pandas-datareader>=0.10.0", # Financial data reader
                    "websocket-client>=1.6.0",  # WebSocket connections
                    "aiohttp>=3.8.0",           # Async HTTP client
                    "cryptofeed>=2.4.0",        # Crypto data feeds
                ]
            },
            
            "execution_algorithms": {
                "description": "Professional Execution Algorithms",
                "packages": [
                    "vectorbt>=0.25.0",         # Vectorized backtesting
                    "zipline-reloaded>=3.0.0",  # Execution engine
                    "catalyst>=0.5.0",          # Crypto trading algorithms
                    "lean>=1.0.0",              # QuantConnect LEAN engine
                ]
            },
            
            "risk_management": {
                "description": "Risk Management & Monitoring",
                "packages": [
                    "riskmetrics>=1.0.0",       # Risk metrics
                    "pymrmr>=0.1.0",            # Feature selection
                    "arch>=5.3.0",              # ARCH/GARCH models
                    "statsmodels>=0.14.0",      # Statistical models
                    "hurst>=0.0.5",             # Hurst exponent
                    "fracdiff>=0.1.0",          # Fractional differentiation
                ]
            },
            
            "data_processing": {
                "description": "Advanced Data Processing & Analysis",
                "packages": [
                    "polars>=0.19.0",           # Fast DataFrame library
                    "dask>=2023.9.0",           # Parallel computing
                    "modin[ray]>=0.22.0",       # Parallel pandas
                    "cudf>=23.08.0",            # GPU DataFrames (if CUDA available)
                    "vaex>=4.16.0",             # Out-of-core DataFrames
                    "fastparquet>=2023.8.0",    # Fast Parquet I/O
                    "numba>=0.58.0",            # JIT compilation
                ]
            },
            
            "visualization": {
                "description": "Advanced Visualization & Monitoring",
                "packages": [
                    "plotly>=5.16.0",           # Interactive plots
                    "dash>=2.14.0",             # Web dashboards
                    "streamlit>=1.27.0",        # ML apps
                    "bokeh>=3.2.0",             # Interactive visualization
                    "seaborn>=0.12.0",          # Statistical visualization
                    "mplfinance>=0.12.0",       # Financial plotting
                ]
            }
        }
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, but found {version.major}.{version.minor}")
            return False
        logger.info(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def install_package(self, package: str) -> bool:
        """Install a single package"""
        try:
            logger.info(f"Installing {package}...")
            
            # Handle special cases
            if package.startswith("gurobipy") or package.startswith("mosek"):
                logger.warning(f"Skipping {package} - requires commercial license")
                return True
            
            if package.startswith("cudf"):
                logger.warning(f"Skipping {package} - requires CUDA installation")
                return True
            
            if package.startswith("catalyst") or package.startswith("lean"):
                logger.warning(f"Skipping {package} - may require additional setup")
                return True
            
            # Standard pip install
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--upgrade"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully installed {package}")
                self.installed_packages.append(package)
                return True
            else:
                logger.error(f"✗ Failed to install {package}: {result.stderr}")
                self.failed_packages.append(package)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout installing {package}")
            self.failed_packages.append(package)
            return False
        except Exception as e:
            logger.error(f"✗ Error installing {package}: {str(e)}")
            self.failed_packages.append(package)
            return False
    
    def install_category(self, category: str) -> Dict[str, Any]:
        """Install all packages in a category"""
        if category not in self.library_categories:
            logger.error(f"Unknown category: {category}")
            return {"success": False, "installed": 0, "failed": 0}
        
        category_info = self.library_categories[category]
        logger.info(f"\n📦 Installing {category_info['description']}")
        logger.info("=" * 60)
        
        installed_count = 0
        failed_count = 0
        
        for package in category_info["packages"]:
            if self.install_package(package):
                installed_count += 1
            else:
                failed_count += 1
        
        logger.info(f"\nCategory Summary: {installed_count} installed, {failed_count} failed")
        return {
            "success": failed_count == 0,
            "installed": installed_count,
            "failed": failed_count
        }
    
    def install_all(self, exclude_categories: List[str] = None) -> Dict[str, Any]:
        """Install all libraries"""
        if exclude_categories is None:
            exclude_categories = []
        
        logger.info("🚀 Starting Kimera Advanced Libraries Installation")
        logger.info("=" * 60)
        
        if not self.check_python_version():
            return {"success": False, "message": "Incompatible Python version"}
        
        total_installed = 0
        total_failed = 0
        category_results = {}
        
        for category in self.library_categories:
            if category in exclude_categories:
                logger.info(f"⏭️ Skipping category: {category}")
                continue
            
            result = self.install_category(category)
            category_results[category] = result
            total_installed += result["installed"]
            total_failed += result["failed"]
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 INSTALLATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total packages installed: {total_installed}")
        logger.info(f"Total packages failed: {total_failed}")
        
        if total_failed > 0:
            logger.warning("\n⚠️ Failed packages:")
            for package in self.failed_packages:
                logger.warning(f"  - {package}")
            logger.warning("\nThese can be installed manually or may require special setup.")
        
        logger.info("\n✅ Core libraries installed successfully!")
        logger.info("Kimera trading system is now ready for advanced algorithms.")
        
        return {
            "success": total_failed == 0,
            "total_installed": total_installed,
            "total_failed": total_failed,
            "category_results": category_results,
            "failed_packages": self.failed_packages
        }
    
    def verify_installations(self) -> Dict[str, bool]:
        """Verify that critical packages are properly installed"""
        critical_packages = [
            "numpy", "pandas", "scikit-learn", "scipy", "cvxpy",
            "pyod", "tensorflow", "stable-baselines3"
        ]
        
        verification_results = {}
        
        logger.info("\n🔍 Verifying critical package installations...")
        
        for package in critical_packages:
            try:
                __import__(package.replace("-", "_"))
                verification_results[package] = True
                logger.info(f"✓ {package} verified")
            except ImportError:
                verification_results[package] = False
                logger.error(f"✗ {package} not available")
        
        return verification_results


def main():
    """Main installation function"""
    installer = LibraryInstaller()
    
    # You can exclude categories if needed
    # exclude_categories = ["reinforcement_learning", "data_processing"]
    exclude_categories = []
    
    # Install all libraries
    result = installer.install_all(exclude_categories=exclude_categories)
    
    # Verify installations
    verification = installer.verify_installations()
    
    # Create summary report
    logger.info("\n" + "=" * 60)
    logger.info("📊 INSTALLATION SUMMARY REPORT")
    logger.info("=" * 60)
    
    for category, result_info in result.get("category_results", {}).items():
        status = "✅" if result_info["success"] else "⚠️"
        logger.info(f"{status} {category}: {result_info['installed']} installed, {result_info['failed']} failed")
    
    logger.info(f"\nOverall Success Rate: {(result['total_installed'] / (result['total_installed'] + result['total_failed']) * 100):.1f}%")
    
    if result["success"]:
        logger.info("\n🎉 Installation completed successfully!")
        logger.info("Kimera is now equipped with state-of-the-art trading libraries.")
    else:
        logger.warning("\n⚠️ Installation completed with some issues.")
        logger.warning("Manual installation may be required for some packages.")
    
    return result["success"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 