import asyncio
from backend.core.therapeutic_intervention_system import TherapeuticInterventionSystem

async def main():
    tis = TherapeuticInterventionSystem()
    alert = {"action": "CREATE_MIRROR_PORTAL", "details": "Demonstration of mirror portal creation"}
    await tis.trigger_mirror_portal_creation(alert)

if __name__ == "__main__":
    asyncio.run(main())
