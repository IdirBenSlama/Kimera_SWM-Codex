import asyncio

from kimera_trading.monitoring.thermodynamic_dash import ThermodynamicDashboard


async def run_monitoring():
    dashboard = ThermodynamicDashboard()
    await dashboard.start_monitoring()


if __name__ == "__main__":
    asyncio.run(run_monitoring())
