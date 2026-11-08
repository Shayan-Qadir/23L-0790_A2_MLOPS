# Task 6: Containerization with Docker, Docker Compose, and CI

## 6.1 Docker Warm-Up & Core Concept Definitions
Demonstrated using script `scripts/task6/docker_warmup.ps1` utilizing `nginx:alpine`:
* Running a container in detached mode (`docker run -d`)
* Port mapping host to container (`-p 8088:80`)
* Mounting local host volume (`-v`)
* Reading runtime logs (`docker logs`)
* Shell execution inside running container (`docker exec`)
* Container and image lifecycle management (`docker ps`, `docker stop`, `docker rm`, `docker rmi`)

### Core Container Concepts
* **Image:** A static, immutable snapshot packaging application source code, system binaries, runtime libraries, and environment variables. Images are composed of read-only stacked layers.
* **Container:** An active, running process encapsulated within an isolated user space. It instantiates an image by attaching a thin read-write container layer on top of the image's immutable base layers.
* **Layer:** An immutable cryptographic filesystem delta created by each directive (e.g., `RUN`, `COPY`, `ADD`) in a `Dockerfile`. Docker caches layers hierarchically; if an instruction's inputs haven't changed, subsequent builds reuse the cached layer instantly.
* **Volume:** A Docker-managed storage mechanism allocated outside the container root filesystem. Volumes allow data (such as PostgreSQL database records) to persist permanently across container restarts, teardowns, and upgrades.
* **Port Mapping:** An iptables/NAT routing bridge that binds a port on the host network interface to a private, isolated listening port inside the container network (e.g., `-p 5000:5000`).

---

## 6.2 Dockerfile Engineering & Layer Optimization
Key architectural decisions in `Dockerfile`:
1. **Lightweight Slim Base:** `python:3.11-slim` provides a minimal Debian environment (~140MB) without the bulk of full development toolchains.
2. **Instruction Ordering & Layer Caching:**
   * `COPY requirements_docker.txt ./requirements.txt` occurs *before* copying application code (`COPY src/`).
   * When Python source code changes during development, Docker reuses the cached layer containing installed packages without re-downloading dependencies, reducing rebuild times from 3 minutes to 2 seconds.
3. **CPU-Only TensorFlow Package (`tensorflow-cpu`):**
   * Eliminates unnecessary GPU/CUDA/cuDNN drivers from container layers.
   * Decreases image size from $>2.8$ GB to $<650$ MB.
4. **Security Principle of Least Privilege:**
   * Dedicated non-root user `appuser` (UID 10001) is created with ownership restricted to `/app`.
   * Execution switches to `USER appuser`, preventing privilege escalation attacks.
5. **Built-in Container Health Check:**
   * Periodically polls `curl -f http://localhost:5000/health`. Container marks itself `unhealthy` if the inference server becomes unresponsive.

---

## 6.3 Multi-Container Compose Architecture
Configured in `docker-compose.yml`:
* **Internal Bridge Network:** `visionguard_network` connects `web` and `db` services via internal DNS resolution (`db:5432`).
* **Closed External Database Port:** The database exposes port `5432` only to the internal Compose network; no port is published to the external host, shielding the audit trail from public internet intrusion.
* **Health Check Coordination:** The `web` service uses `depends_on: db: condition: service_healthy` to guarantee that PostgreSQL has completed initialization before Flask attempts database queries.
* **Decoupled Model Delivery:** Model weights are mounted read-only via `./models/champion:/app/models/champion:ro`. This allows champion model updates without triggering a container rebuild.

---

## 6.6 GitHub Actions Continuous Integration (CI)
* **Workflow Configuration:** `.github/workflows/ci.yml` runs on every push and pull request to `main`.
* **Matrix Strategy:** Executes across Python 3.10 and 3.11 runtimes.
* **Quality Gates:** Enforces Flake8 linting for syntax and complexity, executes Pytest automated unit tests, and verifies Docker image buildability.
* **Secrets Management:** Sensitive credentials (such as `DAGSHUB_TOKEN` or cloud deployment keys) must be stored in GitHub Repository **Settings $\to$ Secrets and Variables $\to$ Actions**, never committed to repository workflow files, ensuring secrets are masked in logs and inaccessible to forks.

