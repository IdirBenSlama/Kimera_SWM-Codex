"""
Test compatibility with existing entropy monitor
"""

from backend.monitoring.entropy_monitor import EntropyMonitor
from backend.core.geoid import GeoidState
import numpy as np

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def test_existing_monitor():
    """Test that existing entropy monitor still works"""
    logger.info("Testing existing entropy monitor compatibility...")
    
    # Test basic entropy monitor
    monitor = EntropyMonitor()
    logger.info('✅ Basic entropy monitor created')

    # Create test geoids
    geoids = []
    for i in range(5):
        geoid = GeoidState(
            geoid_id=f'test_{i}',
            semantic_state={f'feature_{j}': np.random.uniform(0.1, 1.0) for j in range(3)},
            symbolic_state={'state': f'test_{i}'},
            embedding_vector=np.random.uniform(0.1, 1.0, 5).tolist()
        )
        geoids.append(geoid)

    logger.info(f'✅ Created {len(geoids)

    # Test entropy calculation
    vault_info = {'vault_a_scars': 10, 'vault_b_scars': 8}
    measurement = monitor.calculate_system_entropy(geoids, vault_info)

    logger.info(f'✅ Entropy calculation successful:')
    logger.info(f'   Shannon Entropy: {measurement.shannon_entropy:.4f}')
    logger.info(f'   System Complexity: {measurement.system_complexity:.4f}')
    logger.info(f'   Geoid Count: {measurement.geoid_count}')

    logger.info('🎉 Existing entropy monitor works perfectly!')
    return True

if __name__ == "__main__":
    test_existing_monitor()