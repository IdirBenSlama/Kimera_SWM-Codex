"""
SQL Injection Prevention for KIMERA System
Ensures safe database interactions using SQLAlchemy Core
Phase 4, Weeks 12-13: Security Hardening
"""

import logging
from typing import Any, Dict, List

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class SafeQueryBuilder:
    """
    Builds and executes safe, parameterized SQL queries
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def execute_query(
        self, query: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute a safe, parameterized query

        Args:
            query: The SQL query with named bind parameters (e.g., :param_name)
            params: A dictionary of parameters

        Returns:
            A list of result rows as dictionaries
        """
        try:
            # Create bind parameters
            bind_params = [bindparam(key, value) for key, value in params.items()]

            # Create a text clause with bind parameters
            stmt = text(query).bindparams(*bind_params)

            # Execute the query
            result = await self.session.execute(stmt)

            # Fetch results
            return [row._asdict() for row in result.fetchall()]

        except Exception as e:
            logger.error(f"Error executing safe query: {e}")
            raise

    async def get_user_by_username(self, username: str) -> List[Dict[str, Any]]:
        """
        Example of a safe query to get a user by username
        """
        query = "SELECT * FROM users WHERE username = :username"
        params = {"username": username}
        return await self.execute_query(query, params)

    async def search_documents(
        self, search_term: str, user_id: int
    ) -> List[Dict[str, Any]]:
        """
        Example of a safe query with multiple parameters
        """
        query = """
            SELECT * FROM documents 
            WHERE user_id = :user_id AND content LIKE :search_term
        """
        params = {"user_id": user_id, "search_term": f"%{search_term}%"}
        return await self.execute_query(query, params)


# Example of how to use the SafeQueryBuilder
# This would typically be used within your data access layer

# from src.core.database_optimization import get_db_optimization
#
# async def get_user_data(username: str):
#     async with get_db_optimization().optimized_session() as session:
#         query_builder = SafeQueryBuilder(session)
#         user = await query_builder.get_user_by_username(username)
#         return user


# --- Best Practices for Preventing SQL Injection ---

# 1. Always use parameterized queries. Never use string formatting to build queries.
#    BAD:  f"SELECT * FROM users WHERE username = '{username}'"
#    GOOD: "SELECT * FROM users WHERE username = :username"

# 2. Use an ORM (like SQLAlchemy Core or ORM) that automatically handles parameterization.
#    The SafeQueryBuilder above demonstrates the principles of using SQLAlchemy Core.

# 3. Validate and sanitize all user input before it reaches the database layer.
#    This is handled by the RequestValidator in request_hardening.py.

# 4. Use the principle of least privilege for database users.
#    The database user for the application should only have the minimum necessary permissions.

# 5. Regularly scan your code for potential SQL injection vulnerabilities.
#    Tools like Bandit can help with this.
