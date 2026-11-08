#!/usr/bin/env bash
# ==============================================================================
# VisionGuard: Oracle Cloud Infrastructure (OCI) Deployment Automation
# Pathway A: Deploying Docker Compose Stack on an Always Free Ubuntu VM
# Roll Number: 23L-0790 | Project: 23L0790_A2_Ops
# ==============================================================================

set -euo pipefail

echo "============================================================"
echo " 🛡️ VISIONGUARD: OCI ALWAYS FREE DEPLOYMENT SCRIPT"
echo "============================================================"

# 1. System Package Updates & Docker Engine Installation
echo "[*] Step 1: Updating system packages and installing Docker..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release iptables-persistent

# Install Docker GPG key & repository if not present
if ! command -v docker &> /dev/null; then
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "[+] Docker Engine and Docker Compose Plugin installed successfully."
fi

# 2. Firewall Management (Crucial for Oracle Cloud!)
# NOTE: OCI has TWO firewalls:
#   1. Virtual Cloud Network (VCN) Ingress Rule in OCI Console (Port 5000 -> 0.0.0.0/0)
#   2. Local Ubuntu OS firewall (iptables / ufw)
echo "[*] Step 2: Opening port 5000 in OS firewall (iptables)..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 5000 -j ACCEPT
sudo netfilter-persistent save

# 3. Pull or Clone Repository
WORKDIR="$HOME/23L0790_A2_Ops"
if [ ! -d "$WORKDIR" ]; then
    echo "[*] Step 3: Cloning project repository from DagsHub..."
    git clone https://dagshub.com/Shayan-Qadir/23L0790_A2_Ops.git "$WORKDIR"
else
    echo "[*] Step 3: Repository exists at $WORKDIR. Pulling latest commits..."
    cd "$WORKDIR" && git pull origin main
fi

cd "$WORKDIR"

# 4. Configure Environment
if [ ! -f ".env" ]; then
    echo "[*] Step 4: Creating production .env file from .env.example..."
    cp .env.example .env
    # Generate random database password
    RANDOM_PASS=$(openssl rand -hex 16)
    sed -i "s/visionguard_secure_password_2026/$RANDOM_PASS/g" .env
fi

# 5. Build and Launch Multi-Container Stack
echo "[*] Step 5: Building and starting Docker Compose stack..."
docker compose down --remove-orphans || true
docker compose up -d --build

# 6. Verify Service Readiness
echo "[*] Step 6: Verifying service health..."
sleep 10
docker compose ps

echo "[*] Testing /health endpoint..."
curl -s http://localhost:5000/health | jq . || curl -s http://localhost:5000/health

echo "============================================================"
echo " 🚀 VISIONGUARD DEPLOYED SUCCESSFULLY ON ORACLE CLOUD!"
echo " Public Endpoint: http://<YOUR_OCI_PUBLIC_IP>:5000"
echo " Health Check   : http://<YOUR_OCI_PUBLIC_IP>:5000/health"
echo " API Predict    : http://<YOUR_OCI_PUBLIC_IP>:5000/api/predict"
echo "============================================================"

