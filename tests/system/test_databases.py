#!/usr/bin/env python3
"""
Database Connection Test Script for Kimera SWM
Tests PostgreSQL, Neo4j, and Redis connections
"""

import sys
import time
from typing import Any, Dict


def test_postgresql():
    """Test PostgreSQL connection"""
    try:
        import psycopg2

        print("🔍 Testing PostgreSQL connection...")

        # Try connection with proper timeout
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="kimera_swm",
            user="kimera",
            password="kimera_secure_pass_2025",
            connect_timeout=10,
        )

        cursor = conn.cursor()
        cursor.execute("SELECT current_user, current_database(), version();")
        user, db, version = cursor.fetchone()

        print(f"   ✅ PostgreSQL connection successful!")
        print(f"   📊 User: {user}")
        print(f"   📚 Database: {db}")
        print(f"   🔧 Version: {version.split(',')[0]}")

        # Test pgvector extension
        cursor.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';")
        vector_count = cursor.fetchone()[0]
        if vector_count > 0:
            print("   🚀 pgvector extension: ✅ Installed")
        else:
            print("   ⚠️ pgvector extension: Not found")

        cursor.close()
        conn.close()
        return True

    except ImportError:
        print("   ❌ psycopg2 module not installed")
        return False
    except Exception as e:
        print(f"   ❌ PostgreSQL connection failed: {e}")
        return False


def test_neo4j():
    """Test Neo4j connection"""
    try:
        from neo4j import GraphDatabase

        print("\n🔍 Testing Neo4j connection...")

        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "kimera_graph_pass_2025")
        )

        # Test connection with timeout
        with driver.session() as session:
            result = session.run("RETURN 1 as test, apoc.version() as apoc_version")
            record = result.single()

            print("   ✅ Neo4j connection successful!")

            # Check database info
            db_info = session.run(
                """
                CALL dbms.components() 
                YIELD name, versions, edition 
                RETURN name, versions[0] as version, edition
            """
            )

            for record in db_info:
                if record["name"] == "Neo4j Kernel":
                    print(f"   🔧 Version: {record['version']} ({record['edition']})")

            # Check if APOC is available
            try:
                apoc_result = session.run("RETURN apoc.version() as version")
                apoc_version = apoc_result.single()["version"]
                print(f"   🔌 APOC Plugin: ✅ v{apoc_version}")
            except Exception as e:
                logger.error(f"Error in test_databases.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                print("   🔌 APOC Plugin: ⚠️ Not available")

        driver.close()
        return True

    except ImportError:
        print("   ❌ neo4j module not installed")
        return False
    except Exception as e:
        print(f"   ❌ Neo4j connection failed: {e}")
        return False


def test_redis():
    """Test Redis connection"""
    try:
        import redis

        print("\n🔍 Testing Redis connection...")

        r = redis.Redis(
            host="localhost",
            port=6379,
            password="kimera_cache_pass_2025",
            decode_responses=True,
            socket_timeout=10,
        )

        # Test connection
        r.ping()

        info = r.info()
        print(f"   ✅ Redis connection successful!")
        print(f"   🔧 Version: {info['redis_version']}")
        print(f"   💾 Memory: {info['used_memory_human']}")
        print(f"   🔗 Connected clients: {info['connected_clients']}")

        # Test basic operations
        r.set("kimera_test", "connection_test", ex=60)
        test_value = r.get("kimera_test")
        if test_value == "connection_test":
            print("   🧪 Read/Write test: ✅ Passed")
        r.delete("kimera_test")

        return True

    except ImportError:
        print("   ❌ redis module not installed")
        return False
    except Exception as e:
        print(f"   ❌ Redis connection failed: {e}")
        return False


def test_database_schema():
    """Test if we can create basic schema"""
    try:
        import psycopg2

        print("\n🔍 Testing database schema creation...")

        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="kimera_swm",
            user="kimera",
            password="kimera_secure_pass_2025",
        )

        cursor = conn.cursor()

        # Test table creation (simplified)
        test_table_sql = """
        CREATE TABLE IF NOT EXISTS connection_test (
            id SERIAL PRIMARY KEY,
            test_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor.execute(test_table_sql)
        conn.commit()

        # Test insert
        cursor.execute(
            "INSERT INTO connection_test (test_data) VALUES (%s)",
            ("Database connection test",),
        )
        conn.commit()

        # Test select
        cursor.execute("SELECT COUNT(*) FROM connection_test")
        count = cursor.fetchone()[0]

        # Cleanup
        cursor.execute("DROP TABLE connection_test")
        conn.commit()

        print(f"   ✅ Schema operations successful!")
        print(f"   📊 Test records: {count}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"   ❌ Schema test failed: {e}")
        return False


def main():
    """Run all database tests"""
    print("🧪 KIMERA SWM Database Connection Tests")
    print("=" * 50)

    results = {
        "postgresql": test_postgresql(),
        "neo4j": test_neo4j(),
        "redis": test_redis(),
        "schema": test_database_schema(),
    }

    print("\n📊 Test Summary:")
    print("-" * 30)

    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():12} {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All database tests passed! Kimera SWM is ready to start.")
        return 0
    else:
        print("⚠️  Some database tests failed. Check configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
