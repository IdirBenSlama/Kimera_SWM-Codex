import subprocess
import time
import os
import sqlite3
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuration
SQLITE_DB_PATH = "test_mcp.db"

def run_server_test(server_name, test_logic_fn, server_args=None):
    """
    A generic function to start a server and run a test using the MCP client.
    
    :param server_name: The command to start the server (e.g., "mcp-server-sqlite").
    :param test_logic_fn: A function containing the client-side test logic.
    :param server_args: A list of additional arguments for the server command.
    :return: True if the test passes, False otherwise.
    """
    print(f"--- Testing {server_name} ---")
    if server_args is None:
        server_args = []
        
    try:
        # Create server parameters for the MCP client
        server_params = StdioServerParameters(
            command=server_name,
            args=server_args,
            env=None
        )
        
        print(f"Starting server with command: {server_name} {' '.join(server_args)}")
        
        # Run the provided test logic with the server parameters
        test_logic_fn(server_params)
        
        print(f"[{server_name.upper()}] TEST PASSED")
        return True

    except Exception as e:
        print(f"An error occurred during the test for {server_name}: {e}")
        return False
    finally:
        # Cleanup
        if server_name == "mcp-server-sqlite" and os.path.exists(SQLITE_DB_PATH):
            os.remove(SQLITE_DB_PATH)
            print(f"Removed temporary database: {SQLITE_DB_PATH}")
        print("-" * (len(server_name) + 12))
        print("\\n")


def sqlite_test_logic(server_params):
    """Tests the SQLite server by connecting and listing tools."""
    import asyncio
    
    async def test():
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print(f"Connected to SQLite server with {len(tools.tools)} tools")
                
                if tools.tools:
                    print(f"Available tools: {[tool.name for tool in tools.tools]}")
                    
                    # Try to call a tool if available
                    if any(tool.name == "execute_query" for tool in tools.tools):
                        result = await session.call_tool("execute_query", {
                            "query": "SELECT 'Hello from Kimera' as message"
                        })
                        print(f"Tool execution result: {result.content}")
                
                print("SQLite server test completed successfully")
    
    asyncio.run(test())

def fetch_test_logic(server_params):
    """Tests the Fetch server by connecting and listing tools."""
    import asyncio
    
    async def test():
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print(f"Connected to Fetch server with {len(tools.tools)} tools")
                
                if tools.tools:
                    print(f"Available tools: {[tool.name for tool in tools.tools]}")
                    
                    # Try to call a tool if available
                    if any(tool.name == "fetch" for tool in tools.tools):
                        result = await session.call_tool("fetch", {
                            "url": "http://info.cern.ch"
                        })
                        print(f"Fetch result length: {len(str(result.content))} characters")
                
                print("Fetch server test completed successfully")
    
    asyncio.run(test())


if __name__ == "__main__":
    print("Starting verification of Python-based MCP servers...")
    
    results = {}
    
    # Test 1: SQLite Server
    results["SQLite Server"] = run_server_test(
        "mcp-server-sqlite", 
        sqlite_test_logic,
        server_args=["--db-path", SQLITE_DB_PATH]
    )
    
    # Test 2: Fetch Server
    results["Fetch Server"] = run_server_test(
        "mcp-server-fetch",
        fetch_test_logic
    )

    print("--- VERIFICATION SUMMARY ---")
    for server, passed in results.items():
        status = "SUCCESS" if passed else "FAILURE"
        print(f"- {server}: {status}")
    print("----------------------------")
    
    if all(results.values()):
        print("All tested Python MCP servers are working correctly!")
    else:
        print("Some MCP servers failed the verification test.") 