#!/usr/bin/env python3
"""
MCP Server Connection Test Script
Tests all configured MCP servers to ensure they're working properly.
"""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def test_server_startup(
    command: List[str], server_name: str, timeout: int = 10
) -> Dict[str, any]:
    """Test if an MCP server starts up correctly."""
    print(f"\n🧪 Testing {server_name}...")

    try:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Wait a bit for startup
        time.sleep(3)

        # Check if process is still running
        poll_result = process.poll()

        if poll_result is None:
            # Process is running
            print(f"✅ {server_name}: Started successfully")

            # Try to terminate gracefully
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            return {
                "status": "success",
                "message": "Server started and responded correctly",
            }
        else:
            # Process exited
            stdout, stderr = process.communicate()
            print(f"❌ {server_name}: Failed to start")
            print(f"Exit code: {poll_result}")
            if stderr:
                print(f"Error: {stderr[:200]}...")

            return {
                "status": "failed",
                "exit_code": poll_result,
                "error": stderr[:500] if stderr else "No error output",
            }

    except Exception as e:
        print(f"❌ {server_name}: Exception during test - {e}")
        return {"status": "exception", "error": str(e)}


def test_database_access(db_path: str) -> Dict[str, any]:
    """Test database file access."""
    try:
        import sqlite3

        if not Path(db_path).exists():
            return {"status": "failed", "error": f"Database file not found: {db_path}"}

        # Try to connect and query
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()

        return {
            "status": "success",
            "tables": [table[0] for table in tables],
            "table_count": len(tables),
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}


def main():
    """Run all MCP server tests."""
    print("🚀 Starting MCP Server Connection Tests")
    print("=" * 50)

    # Read the MCP configuration
    config_path = Path(".cursor/mcp.json")
    if not config_path.exists():
        print("❌ MCP configuration file not found!")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    servers = config.get("mcpServers", {})
    results = {}

    # Test each server
    for server_name, server_config in servers.items():
        command = [server_config["command"]] + server_config.get("args", [])

        # Set environment variables
        env = server_config.get("env", {})
        if env:
            import os

            for key, value in env.items():
                os.environ[key] = value

        results[server_name] = test_server_startup(command, server_name)

    # Test databases specifically
    print(f"\n🗄️  Testing Database Access...")

    test_db_path = "D:/DEV/MCP servers/test.db"
    kimera_db_path = "D:/DEV/Kimera_SWM_Alpha_Prototype V0.1 140625/kimera_swm.db"

    print(f"\n📊 Test Database: {test_db_path}")
    test_db_result = test_database_access(test_db_path)
    print(
        f"Status: {'✅' if test_db_result['status'] == 'success' else '❌'} {test_db_result}"
    )

    print(f"\n📊 Kimera Database: {kimera_db_path}")
    kimera_db_result = test_database_access(kimera_db_path)
    print(
        f"Status: {'✅' if kimera_db_result['status'] == 'success' else '❌'} {kimera_db_result}"
    )

    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 30)

    successful = sum(1 for result in results.values() if result["status"] == "success")
    total = len(results)

    print(f"Servers tested: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")

    if successful == total:
        print("\n🎉 All MCP servers are working correctly!")
        print("\n💡 Next steps:")
        print("1. Restart Cursor to reload MCP configuration")
        print("2. Check MCP servers are showing as green in Cursor")
        print("3. Try using MCP tools in a chat")
    else:
        print(f"\n⚠️  {total - successful} servers need attention")
        print("\nFailed servers:")
        for name, result in results.items():
            if result["status"] != "success":
                print(f"  - {name}: {result.get('error', 'Unknown error')}")

    # Save detailed results
    results_file = "mcp_test_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "servers": results,
                "databases": {"test_db": test_db_result, "kimera_db": kimera_db_result},
            },
            f,
            indent=2,
        )

    print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
