# PostgreSQL Installation and Setup
# Run as Administrator

# Download and install PostgreSQL
Write-Host "📦 Installing PostgreSQL..."

# Using Chocolatey (recommended)
if (Get-Command choco -ErrorAction SilentlyContinue) {
    choco install postgresql13 -y
} else {
    Write-Host "⚠️ Chocolatey not found. Please install PostgreSQL manually from:"
    Write-Host "   https://www.postgresql.org/download/windows/"
    Write-Host "   Version: 13 or higher"
}

# Wait for service to start
Start-Sleep -Seconds 10

# Create Kimera database and user
Write-Host "🔧 Setting up Kimera database..."

# Connect to PostgreSQL and create database
$env:PGPASSWORD="postgres"
psql -U postgres -c "CREATE DATABASE kimera_swm;"
psql -U postgres -c "CREATE USER kimera_user WITH PASSWORD 'kimera_secure_pass';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE kimera_swm TO kimera_user;"

Write-Host "✅ PostgreSQL setup complete!"
