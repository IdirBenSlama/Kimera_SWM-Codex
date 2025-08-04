# Neo4j Installation and Setup
# Run as Administrator

Write-Host "Installing Neo4j..."

# Using Chocolatey
if (Get-Command choco -ErrorAction SilentlyContinue) {
    choco install neo4j-community -y
} else {
    Write-Host "⚠️ Chocolatey not found. Please install Neo4j manually from:"
    Write-Host "   https://neo4j.com/download/"
    Write-Host "   Version: Community 4.4 or higher"
}

# Start Neo4j service
Write-Host "🚀 Starting Neo4j service..."
Start-Service -Name "neo4j"

# Set initial password
Write-Host "🔧 Setting up Neo4j authentication..."
neo4j-admin set-initial-password kimera_neo4j

Write-Host "✅ Neo4j setup complete!"
Write-Host "🌐 Access Neo4j Browser at: http://localhost:7474"
