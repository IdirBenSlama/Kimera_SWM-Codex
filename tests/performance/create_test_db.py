import sqlite3

# Create test database
conn = sqlite3.connect("test.db")
cursor = conn.cursor()

# Create test table
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS test (
    id INTEGER PRIMARY KEY,
    name TEXT
)
"""
)

# Insert test data
cursor.execute('INSERT OR IGNORE INTO test (id, name) VALUES (1, "Test Entry")')
cursor.execute('INSERT OR IGNORE INTO test (id, name) VALUES (2, "Kimera MCP Test")')

conn.commit()
conn.close()

print("Test database created successfully with sample data!")
