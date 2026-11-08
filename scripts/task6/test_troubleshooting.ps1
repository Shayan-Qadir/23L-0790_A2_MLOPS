Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " TASK 6.5: COMPOSE OPERATIONS & TROUBLESHOOTING DEMO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Normal Service Verification
Write-Host "`n[*] 1. Checking running services..." -ForegroundColor Yellow
docker compose ps

# 2. Demonstrate container operations: logs, shell, restart
Write-Host "`n[*] 2. Reading logs from web and db services..." -ForegroundColor Yellow
docker compose logs --tail=10 web
docker compose logs --tail=10 db

Write-Host "`n[*] 3. Executing diagnostic command inside web container (docker exec)..." -ForegroundColor Yellow
docker compose exec web python -c "import flask; print('Flask version:', flask.__version__)"
docker compose exec web whoami

# 3. Purposeful Break Test: Misconfigure Database Password
Write-Host "`n[*] 4. Simulating failure: Breaking database password in environment..." -ForegroundColor Red
$envBackup = Get-Content .env
# Inject incorrect password
(Get-Content .env) -replace "POSTGRES_PASSWORD=.*", "POSTGRES_PASSWORD=INTENTIONALLY_WRONG_PASSWORD_123" | Set-Content .env

Write-Host "[!] Restarting web container with broken credentials..." -ForegroundColor Red
docker compose up -d --force-recreate web

Start-Sleep -Seconds 5
Write-Host "`n============================================================" -ForegroundColor Red
Write-Host " DIAGNOSTIC LOG EXCERPT SHOWING FAILURE" -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Red
docker compose logs --tail=20 web

# 4. Fix: Restore Correct Password
Write-Host "`n[*] 5. Applying Fix: Restoring correct database credentials in .env..." -ForegroundColor Green
$envBackup | Set-Content .env
docker compose up -d --force-recreate web

Start-Sleep -Seconds 5
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " VERIFYING SYSTEM RECOVERY (FIXED STATE)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
docker compose ps
docker compose logs --tail=10 web

Write-Host "`n[+] Troubleshooting experiment completed successfully!" -ForegroundColor Green

