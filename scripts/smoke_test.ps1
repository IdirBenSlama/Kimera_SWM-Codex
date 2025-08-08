Param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$TimeoutSec = 10
)

$ErrorActionPreference = "Stop"

function Test-Endpoint($path) {
    $url = "$BaseUrl$path"
    try {
        $resp = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec $TimeoutSec -UseBasicParsing
        return @{ ok = $true; status = $resp.StatusCode; url = $url }
    } catch {
        return @{ ok = $false; status = 0; url = $url; err = $_.Exception.Message }
    }
}

$checks = @(
    "/openapi.json",
    "/docs",
    "/system/status"
)

$passed = 0
$failed = 0
foreach ($path in $checks) {
    $r = Test-Endpoint $path
    if ($r.ok) {
        Write-Host "[PASS] $($r.url) ($($r.status))" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "[FAIL] $($r.url) ($($r.err))" -ForegroundColor Red
        $failed++
    }
}

Write-Host "---" -ForegroundColor DarkGray
Write-Host "Passed: $passed  Failed: $failed" -ForegroundColor Cyan
if ($failed -gt 0) { exit 1 } else { exit 0 }
