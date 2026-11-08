Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " TASK 6.1: DOCKER WARM-UP DEMONSTRATION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Pull ready-made lightweight image
Write-Host "`n[*] 1. Pulling ready-made image (nginx:alpine)..." -ForegroundColor Yellow
docker pull nginx:alpine

# 2. Run container with mapped port and volume mount
Write-Host "`n[*] 2. Running container with port mapping (8088:80) and volume mount..." -ForegroundColor Yellow
$tempVol = New-Item -ItemType Directory -Path "$PWD/temp_docker_vol" -Force
"<h1>VisionGuard Docker Warm-Up Successful!</h1>" | Out-File -FilePath "$tempVol/index.html" -Encoding utf8
docker run -d --name warmup_nginx -p 8088:80 -v "${tempVol}:/usr/share/nginx/html:ro" nginx:alpine

# 3. Read container logs
Write-Host "`n[*] 3. Reading container logs..." -ForegroundColor Yellow
docker logs warmup_nginx

# 4. Open and execute command inside container
Write-Host "`n[*] 4. Executing command inside container shell (docker exec)..." -ForegroundColor Yellow
docker exec warmup_nginx uname -a
docker exec warmup_nginx whoami

# 5. List running containers and images
Write-Host "`n[*] 5. Listing containers and images..." -ForegroundColor Yellow
docker ps -f name=warmup_nginx
docker images nginx:alpine

# 6. Stop and remove container, clean up image
Write-Host "`n[*] 6. Stopping and removing warmup container..." -ForegroundColor Yellow
docker stop warmup_nginx
docker rm warmup_nginx
docker rmi nginx:alpine
Remove-Item -Path $tempVol -Recurse -Force

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " DOCKER WARM-UP COMPLETED FLINTLESSLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

