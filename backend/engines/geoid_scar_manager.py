"""
Geoid SCAR Manager
==================
Manages Semantic Contextual Anomaly Representations (SCARs) and Geoids.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from ..utils.config import get_api_settings
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class GeoidScarManager:
    """Manages Geoids and SCARs in the Kimera system"""
    
    def __init__(self):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.geoids: Dict[str, Dict[str, Any]] = {}
        self.scars: Dict[str, Dict[str, Any]] = {}
        logger.info("GeoidScarManager initialized")
        
    async def create_geoid(self, data: Dict[str, Any]) -> str:
        """Create a new geoid"""
        geoid_id = str(uuid.uuid4())
        self.geoids[geoid_id] = {
            "id": geoid_id,
            "data": data,
            "created_at": datetime.utcnow().isoformat(),
            "scars": []
        }
        logger.info(f"Created geoid: {geoid_id}")
        return geoid_id
        
    async def create_scar(self, geoid_ids: List[str], reason: str) -> str:
        """Create a new SCAR"""
        scar_id = str(uuid.uuid4())
        self.scars[scar_id] = {
            "id": scar_id,
            "geoid_ids": geoid_ids,
            "reason": reason,
            "created_at": datetime.utcnow().isoformat(),
            "resolved": False
        }
        
        # Link SCAR to geoids
        for geoid_id in geoid_ids:
            if geoid_id in self.geoids:
                self.geoids[geoid_id]["scars"].append(scar_id)
                
        logger.info(f"Created SCAR: {scar_id}")
        return scar_id
        
    async def get_geoid(self, geoid_id: str) -> Optional[Dict[str, Any]]:
        """Get a geoid by ID"""
        return self.geoids.get(geoid_id)
        
    async def get_scar(self, scar_id: str) -> Optional[Dict[str, Any]]:
        """Get a SCAR by ID"""
        return self.scars.get(scar_id)
        
    async def list_geoids(self) -> List[Dict[str, Any]]:
        """List all geoids"""
        return list(self.geoids.values())
        
    async def list_scars(self) -> List[Dict[str, Any]]:
        """List all SCARs"""
        return list(self.scars.values())
        
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            "total_geoids": len(self.geoids),
            "total_scars": len(self.scars),
            "unresolved_scars": sum(1 for s in self.scars.values() if not s["resolved"])
        }
