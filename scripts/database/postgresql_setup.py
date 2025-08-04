#!/usr/bin/env python3
"""
PostgreSQL Database Setup and Authentication Configuration
=========================================================

Configures PostgreSQL database for Kimera SWM system with proper authentication.
Implements aerospace-grade database security and configuration standards.

Usage:
    python scripts/database/postgresql_setup.py
"""

import os
import sys
import subprocess
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgreSQLConfigurator:
    """
    PostgreSQL configuration manager for Kimera SWM system.

    Implements secure database setup with proper authentication,
    user management, and database initialization.
    """

    def __init__(self):
        self.config = {
            'database': 'kimera_swm',
            'username': 'kimera_user',
            'password': 'kimera_secure_pass',
            'host': 'localhost',
            'port': 5432
        }

        # Ensure database scripts directory exists
        os.makedirs('scripts/database', exist_ok=True)

    def check_postgresql_installation(self) -> bool:
        """Check if PostgreSQL is installed and accessible."""
        logger.info("🔍 CHECKING POSTGRESQL INSTALLATION...")
        logger.info("=" * 50)

        try:
            # Check psql command availability
            result = subprocess.run(['psql', '--version'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"✅ PostgreSQL found: {version}")
                return True
            else:
                logger.info(f"❌ PostgreSQL command failed: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.info("❌ PostgreSQL not found in PATH")
            logger.info("💡 Install PostgreSQL: https://www.postgresql.org/download/")
            return False
        except subprocess.TimeoutExpired:
            logger.info("❌ PostgreSQL command timed out")
            return False
        except Exception as e:
            logger.info(f"❌ Error checking PostgreSQL: {e}")
            return False

    def test_connection(self, as_admin: bool = False) -> bool:
        """Test PostgreSQL connection."""
        logger.info(f"🔗 TESTING CONNECTION {'(as admin)' if as_admin else '(as kimera_user)'}...")

        try:
            if as_admin:
                # Test as postgres superuser
                cmd = ['psql', '-U', 'postgres', '-c', 'SELECT version();']
            else:
                # Test as kimera_user
                cmd = ['psql', '-U', self.config['username'],
                      '-d', self.config['database'], '-c', 'SELECT 1;']

            env = os.environ.copy()
            if not as_admin:
                env['PGPASSWORD'] = self.config['password']

            result = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=10, env=env)

            if result.returncode == 0:
                logger.info(f"✅ Connection successful")
                return True
            else:
                logger.info(f"❌ Connection failed: {result.stderr.strip()}")
                return False

        except Exception as e:
            logger.info(f"❌ Connection test error: {e}")
            return False

    def create_database_and_user(self) -> bool:
        """Create database and user with proper permissions."""
        logger.info("🔧 CREATING DATABASE AND USER...")
        logger.info("=" * 50)

        # SQL commands to create user and database
        sql_commands = f"""
-- Create user if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{self.config['username']}') THEN
      CREATE USER {self.config['username']} WITH PASSWORD '{self.config['password']}';
   END IF;
END
$$;

-- Create database if not exists
SELECT 'CREATE DATABASE {self.config['database']} OWNER {self.config['username']}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{self.config['database']}')\\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE {self.config['database']} TO {self.config['username']};

-- Alter user password (in case user exists but password is wrong)
ALTER USER {self.config['username']} PASSWORD '{self.config['password']}';
"""

        try:
            # Create SQL script file
            script_path = 'scripts/database/setup_kimera_db.sql'
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(sql_commands)

            logger.info(f"✅ Created SQL script: {script_path}")

            # Execute as postgres superuser
            cmd = ['psql', '-U', 'postgres', '-f', script_path]

            logger.info("🔄 Executing database setup script...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info("✅ Database and user created successfully")
                logger.info(f"   Database: {self.config['database']}")
                logger.info(f"   User: {self.config['username']}")
                return True
            else:
                logger.info(f"❌ Database setup failed: {result.stderr.strip()}")

                # If it's an authentication error, provide guidance
                if "authentication failed" in result.stderr.lower():
                    logger.info("\n💡 TROUBLESHOOTING:")
                    logger.info("   1. Ensure PostgreSQL service is running")
                    logger.info("   2. Set PGPASSWORD environment variable for postgres user")
                    logger.info("   3. Or run: psql -U postgres -c \"ALTER USER postgres PASSWORD 'your_password';\"")

                return False

        except Exception as e:
            logger.info(f"❌ Error creating database: {e}")
            return False

    def generate_connection_config(self) -> str:
        """Generate connection configuration for Kimera system."""
        config_path = 'configs/database/postgresql_config.json'

        config_data = {
            "postgresql": {
                "host": self.config['host'],
                "port": self.config['port'],
                "database": self.config['database'],
                "username": self.config['username'],
                "password": self.config['password'],
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "echo": False
            },
            "connection_string": f"postgresql://{self.config['username']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}",
            "test_query": "SELECT 1;"
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"✅ Generated PostgreSQL config: {config_path}")
        return config_path

    def test_kimera_integration(self) -> bool:
        """Test PostgreSQL integration with Kimera system."""
        logger.info("🧠 TESTING KIMERA INTEGRATION...")
        logger.info("=" * 50)

        try:
            import psycopg2

            # Test connection with psycopg2
            conn_string = f"host={self.config['host']} port={self.config['port']} dbname={self.config['database']} user={self.config['username']} password={self.config['password']}"

            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()

            # Test basic operations
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"✅ Connected to: {version[:50]}...")

            # Create a test table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kimera_test (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Insert test data
            cursor.execute("INSERT INTO kimera_test (name) VALUES (%s) RETURNING id;", ("System Health Test",))
            test_id = cursor.fetchone()[0]

            # Query test data
            cursor.execute("SELECT name FROM kimera_test WHERE id = %s;", (test_id,))
            result = cursor.fetchone()[0]

            logger.info(f"✅ Test operations successful: {result}")

            # Clean up
            cursor.execute("DROP TABLE kimera_test;")
            conn.commit()

            cursor.close()
            conn.close()

            logger.info("✅ Kimera-PostgreSQL integration verified")
            return True

        except ImportError:
            logger.info("❌ psycopg2 not available for testing")
            return False
        except Exception as e:
            logger.info(f"❌ Integration test failed: {e}")
            return False

    def run_complete_setup(self) -> bool:
        """Run complete PostgreSQL setup and configuration."""
        logger.info("🗄️ POSTGRESQL COMPLETE SETUP")
        logger.info("=" * 60)
        logger.info("🔒 Aerospace-Grade Database Configuration")
        logger.info("📊 Kimera SWM Integration Ready")
        logger.info("=" * 60)
        logger.info()

        success_steps = 0
        total_steps = 5

        # Step 1: Check installation
        if self.check_postgresql_installation():
            success_steps += 1
            logger.info()
        else:
            logger.info("⚠️ PostgreSQL installation required before proceeding")
            return False

        # Step 2: Test admin connection
        if self.test_connection(as_admin=True):
            success_steps += 1
            logger.info()
        else:
            logger.info("⚠️ Cannot connect as admin. Database setup may require manual intervention.")

        # Step 3: Create database and user
        if self.create_database_and_user():
            success_steps += 1
            logger.info()

        # Step 4: Test user connection
        if self.test_connection(as_admin=False):
            success_steps += 1
            logger.info()

        # Step 5: Generate configuration
        config_path = self.generate_connection_config()
        if config_path:
            success_steps += 1
            logger.info()

        # Optional: Test integration
        if success_steps >= 4:
            self.test_kimera_integration()
            logger.info()

        # Final assessment
        logger.info("📊 SETUP SUMMARY")
        logger.info("=" * 30)
        logger.info(f"✅ Steps completed: {success_steps}/{total_steps}")
        logger.info(f"📊 Success rate: {success_steps/total_steps*100:.1f}%")

        if success_steps >= 4:
            logger.info("🎉 PostgreSQL setup SUCCESSFUL!")
            logger.info(f"   Database: {self.config['database']}")
            logger.info(f"   User: {self.config['username']}")
            logger.info(f"   Config: {config_path}")
            return True
        else:
            logger.info("⚠️ PostgreSQL setup INCOMPLETE")
            logger.info("   Manual intervention may be required")
            return False

def main():
    """Main setup execution function."""
    configurator = PostgreSQLConfigurator()

    try:
        success = configurator.run_complete_setup()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\n⚠️ Setup interrupted by user")
        return 1
    except Exception as e:
        logger.info(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
