"""Test runner for Kimera trading module"""
import pytest
from pathlib import Path

def run_tests():
    """Run all trading module tests and generate report"""
    test_dir = Path(__file__).parent
    report_dir = test_dir.parent.parent / "test_reports" / "trading"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Use simple counter for report names
    report_count = len(list(report_dir.glob("junit_report_*.xml")))
    junit_path = report_dir / f"junit_report_{report_count + 1}.xml"
    
    # Run basic tests first (without coverage)
    pytest_args = [
        str(test_dir),
        "-v",
        "--junitxml",
        str(junit_path),
        "--log-level=INFO",
        "--log-file",
        str(report_dir / "pytest.log"),
        "--durations=10"
    ]
    
    try:
        exit_code = pytest.main(pytest_args)
        # Generate risk assessment report if tests pass
        if exit_code == 0:
            from backend.trading.risk.cognitive_risk_manager import generate_risk_report
            generate_risk_report(report_dir)
        return exit_code
    except Exception as e:
        print(f"Test runner failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    exit(exit_code)
