@echo off
echo Installing remaining MCP servers for Kimera development...
echo.

REM Install TypeScript globally
echo Installing TypeScript globally...
npm install -g typescript
if %errorlevel% neq 0 (
    echo Failed to install TypeScript globally, trying locally...
    cd official-servers
    npm install typescript --save-dev
    cd ..
)

REM Build official servers
echo Building official MCP servers...
cd official-servers
npm run build
if %errorlevel% neq 0 (
    echo Build failed, trying individual builds...
    cd src\filesystem
    npx tsc
    cd ..\memory
    npx tsc
    cd ..\git
    npx tsc
    cd ..\sequentialthinking
    npx tsc
    cd ..\time
    npx tsc
    cd ..\..
)
cd ..

REM Install additional Python-based servers
echo Installing additional Python MCP servers...
pip install mcp-server-postgres
pip install fastmcp

REM Install community servers via npm (when network is available)
echo Installing community MCP servers...
npm install -g @anaisbetts/mcp-installer

REM Create test database
echo Creating test SQLite database...
python -c "import sqlite3; conn = sqlite3.connect('test.db'); conn.execute('CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)'); conn.execute('INSERT OR IGNORE INTO test (id, name) VALUES (1, \"Test Entry\")'); conn.commit(); conn.close()"

echo.
echo Installation complete!
echo.
echo Configuration file created: claude_desktop_config.json
echo Copy this file to: %APPDATA%\Claude Desktop\claude_desktop_config.json
echo.
echo To test the servers individually:
echo python -m mcp_server_fetch
echo python -m mcp_server_sqlite --db-path "test.db"
echo.
pause 