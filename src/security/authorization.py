"""
Authorization Module
====================

Implements role-based and attribute-based access control.
Based on NIST RBAC and ABAC models.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""

    # Read permissions
    READ_PUBLIC = "read:public"
    READ_PRIVATE = "read:private"
    READ_SYSTEM = "read:system"

    # Write permissions
    WRITE_PUBLIC = "write:public"
    WRITE_PRIVATE = "write:private"
    WRITE_SYSTEM = "write:system"

    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_SECURITY = "admin:security"

    # Special permissions
    EXECUTE_COGNITIVE = "execute:cognitive"
    MODIFY_GOVERNANCE = "modify:governance"
    ACCESS_VAULT = "access:vault"


class Role(Enum):
    """User roles."""

    GUEST = "guest"
    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"
    SECURITY_ADMIN = "security_admin"
    SYSTEM = "system"


@dataclass
class AccessContext:
    """Auto-generated class."""
    pass
    """Context for access control decisions."""

    user_id: str
    roles: Set[Role]
    attributes: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Resource:
    """Auto-generated class."""
    pass
    """Resource being accessed."""

    id: str
    type: str
    owner: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    sensitivity_level: int = 0  # 0-5, higher is more sensitive
class Authorizer:
    """Auto-generated class."""
    pass
    """
    Authorization engine with RBAC and ABAC support.
    """

    def __init__(self):
        self.role_permissions = self._init_role_permissions()
        self.resource_policies = {}
        self.access_log = []

    def _init_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role-permission mappings."""
        return {
            Role.GUEST: {Permission.READ_PUBLIC},
            Role.USER: {
                Permission.READ_PUBLIC,
                Permission.READ_PRIVATE,
                Permission.WRITE_PUBLIC,
                Permission.WRITE_PRIVATE,
                Permission.EXECUTE_COGNITIVE,
            },
            Role.OPERATOR: {
                Permission.READ_PUBLIC,
                Permission.READ_PRIVATE,
                Permission.READ_SYSTEM,
                Permission.WRITE_PUBLIC,
                Permission.WRITE_PRIVATE,
                Permission.EXECUTE_COGNITIVE,
                Permission.ACCESS_VAULT,
            },
            Role.ADMIN: {
                Permission.READ_PUBLIC,
                Permission.READ_PRIVATE,
                Permission.READ_SYSTEM,
                Permission.WRITE_PUBLIC,
                Permission.WRITE_PRIVATE,
                Permission.WRITE_SYSTEM,
                Permission.ADMIN_USERS,
                Permission.EXECUTE_COGNITIVE,
                Permission.ACCESS_VAULT,
                Permission.MODIFY_GOVERNANCE,
            },
            Role.SECURITY_ADMIN: {
                Permission.READ_PUBLIC,
                Permission.READ_PRIVATE,
                Permission.READ_SYSTEM,
                Permission.ADMIN_USERS,
                Permission.ADMIN_SECURITY,
                Permission.ADMIN_SYSTEM,
            },
            Role.SYSTEM: set(Permission),  # All permissions
        }

    def authorize(
        self, context: AccessContext, resource: Resource, permission: Permission
    ) -> bool:
        """
        Authorize access to a resource.

        Args:
            context: Access context with user info
            resource: Resource being accessed
            permission: Required permission

        Returns:
            True if access is granted
        """
        # Log access attempt
        self._log_access_attempt(context, resource, permission)

        # System role bypasses all checks
        if Role.SYSTEM in context.roles:
            return True

        # Check RBAC first
        if not self._check_rbac(context.roles, permission):
            logger.info(
                f"RBAC denied: user {context.user_id} lacks permission {permission.value}"
            )
            return False

        # Check ABAC policies
        if not self._check_abac(context, resource, permission):
            logger.info(
                f"ABAC denied: user {context.user_id} for resource {resource.id}"
            )
            return False

        # Check resource sensitivity
        if not self._check_sensitivity(context, resource):
            logger.info(
                f"Sensitivity check failed: user {context.user_id} for resource {resource.id}"
            )
            return False

        # Check ownership for write operations
        if permission in [Permission.WRITE_PRIVATE, Permission.WRITE_PUBLIC]:
            if not self._check_ownership(context, resource):
                logger.info(
                    f"Ownership check failed: user {context.user_id} for resource {resource.id}"
                )
                return False

        logger.debug(
            f"Access granted: user {context.user_id} permission {permission.value} "
            f"for resource {resource.id}"
        )
        return True

    def _check_rbac(self, roles: Set[Role], permission: Permission) -> bool:
        """Check role-based access control."""
        for role in roles:
            if permission in self.role_permissions.get(role, set()):
                return True
        return False

    def _check_abac(
        self, context: AccessContext, resource: Resource, permission: Permission
    ) -> bool:
        """Check attribute-based access control."""
        # Get resource-specific policies
        policies = self.resource_policies.get(resource.type, [])

        # If no policies, default to allow (after RBAC check)
        if not policies:
            return True

        # Evaluate each policy
        for policy in policies:
            if not policy(context, resource, permission):
                return False

        return True

    def _check_sensitivity(self, context: AccessContext, resource: Resource) -> bool:
        """Check if user can access resource based on sensitivity."""
        # Map roles to max sensitivity levels
        max_sensitivity = {
            Role.GUEST: 0,
            Role.USER: 2,
            Role.OPERATOR: 3,
            Role.ADMIN: 4,
            Role.SECURITY_ADMIN: 5,
            Role.SYSTEM: 5,
        }

        # Get user's max allowed sensitivity
        user_max = max(max_sensitivity.get(role, 0) for role in context.roles)

        return resource.sensitivity_level <= user_max

    def _check_ownership(self, context: AccessContext, resource: Resource) -> bool:
        """Check resource ownership."""
        # Admins can modify any resource
        if Role.ADMIN in context.roles:
            return True

        # Check direct ownership
        return resource.owner == context.user_id

    def _log_access_attempt(
        self, context: AccessContext, resource: Resource, permission: Permission
    ):
        """Log access attempt for audit."""
        self.access_log.append(
            {
                "timestamp": datetime.utcnow(),
                "user_id": context.user_id,
                "resource_id": resource.id,
                "resource_type": resource.type,
                "permission": permission.value,
                "roles": [r.value for r in context.roles],
                "session_id": context.session_id,
                "ip_address": context.ip_address,
            }
        )

        # Keep log size bounded
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]

    def add_resource_policy(self, resource_type: str, policy_func: callable):
        """Add an ABAC policy for a resource type."""
        if resource_type not in self.resource_policies:
            self.resource_policies[resource_type] = []

        self.resource_policies[resource_type].append(policy_func)
        logger.info(f"Added policy for resource type: {resource_type}")

    def grant_permission(self, role: Role, permission: Permission):
        """Grant a permission to a role."""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()

        self.role_permissions[role].add(permission)
        logger.info(f"Granted {permission.value} to role {role.value}")

    def revoke_permission(self, role: Role, permission: Permission):
        """Revoke a permission from a role."""
        if role in self.role_permissions:
            self.role_permissions[role].discard(permission)
            logger.info(f"Revoked {permission.value} from role {role.value}")

    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role."""
        return self.role_permissions.get(role, set()).copy()

    def get_access_log(
        self,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get filtered access log."""
        filtered_log = self.access_log

        if user_id:
            filtered_log = [
                entry for entry in filtered_log if entry["user_id"] == user_id
            ]

        if resource_id:
            filtered_log = [
                entry for entry in filtered_log if entry["resource_id"] == resource_id
            ]

        return filtered_log[-limit:]


# Example ABAC policies
def time_based_policy(
    context: AccessContext, resource: Resource, permission: Permission
) -> bool:
    """Example: Restrict access during certain hours."""
    current_hour = datetime.utcnow().hour

    # Allow access only during business hours (9 AM - 6 PM UTC)
    if 9 <= current_hour < 18:
        return True

    # Admins can access anytime
    return Role.ADMIN in context.roles


def location_based_policy(
    context: AccessContext, resource: Resource, permission: Permission
) -> bool:
    """Example: Restrict access based on IP location."""
    # In production, this would check IP geolocation
    trusted_ips = ["192.168.1.0/24", "10.0.0.0/8"]

    # Simplified check
    if context.ip_address and context.ip_address.startswith(("192.168.", "10.")):
        return True

    # Security admins can access from anywhere
    return Role.SECURITY_ADMIN in context.roles
