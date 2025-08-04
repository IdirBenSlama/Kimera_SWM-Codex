# Prometheus Installation and Setup
# Run as Administrator

Write-Host "📦 Installing Prometheus..."

# Create Prometheus directory
$prometheusDir = "C:\Prometheus"
New-Item -ItemType Directory -Force -Path $prometheusDir

# Download Prometheus
$prometheusVersion = "2.45.0"
$prometheusUrl = "https://github.com/prometheus/prometheus/releases/download/v$prometheusVersion/prometheus-$prometheusVersion.windows-amd64.zip"
$prometheusZip = "$prometheusDir\prometheus.zip"

Invoke-WebRequest -Uri $prometheusUrl -OutFile $prometheusZip
Expand-Archive -Path $prometheusZip -DestinationPath $prometheusDir -Force

# Create Prometheus configuration
$configContent = @"
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kimera-swm'
    static_configs:
      - targets: ['localhost:8001']
        labels:
          service: 'kimera-api'

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"@

Set-Content -Path "$prometheusDir\prometheus.yml" -Value $configContent

Write-Host "✅ Prometheus setup complete!"
Write-Host "🚀 To start Prometheus: cd $prometheusDir && .\prometheus.exe"
