import pytest
from kimera_trading.core.consciousness import ConsciousnessStateManager

@pytest.mark.asyncio
async def test_consciousness_state_manager():
    manager = ConsciousnessStateManager()
    await manager.calibrate(None)
    assert manager.current_state.level == 0.2
