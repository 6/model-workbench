#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Hetzner Cloud Server Bootstrap Script
# =============================================================================
# Idempotent setup for: Langfuse, Loki, Grafana, MLflow, LiteLLM, Prometheus
# Run as root on a fresh Ubuntu 24.04 server after Terraform provisioning.
#
# Usage:
#   DOMAIN=example.com ./setup.sh
#   # or with all options:
#   DOMAIN=example.com ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... ./setup.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration (override via environment)
# -----------------------------------------------------------------------------
DOMAIN="${DOMAIN:-}"
BASE_DIR="${BASE_DIR:-/opt/services}"
TIMEZONE="${TIMEZONE:-UTC}"

# LLM API keys for LiteLLM proxy (optional, can add later)
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# Retention settings
LOKI_RETENTION_DAYS="${LOKI_RETENTION_DAYS:-14}"
PROMETHEUS_RETENTION_DAYS="${PROMETHEUS_RETENTION_DAYS:-14}"
MINIO_RETENTION_DAYS="${MINIO_RETENTION_DAYS:-30}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log() { echo -e "\n==> $*\n"; }
error() { echo -e "\nERROR: $*\n" >&2; exit 1; }

generate_password() {
  openssl rand -base64 24 | tr -d '\n' | tr '/+' 'ab'
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    error "This script must be run as root"
  fi
}

require_domain() {
  if [[ -z "${DOMAIN}" ]]; then
    error "DOMAIN environment variable is required. Example: DOMAIN=example.com ./setup.sh"
  fi
}

# -----------------------------------------------------------------------------
# System Hardening
# -----------------------------------------------------------------------------
harden_system() {
  log "Configuring system hardening..."

  # Set timezone
  timedatectl set-timezone "${TIMEZONE}"

  # Update packages
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

  # Install essentials
  apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    unattended-upgrades \
    fail2ban \
    ufw \
    jq \
    htop \
    ncdu

  # Configure unattended-upgrades (security only)
  cat >/etc/apt/apt.conf.d/50unattended-upgrades <<'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

  cat >/etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

  systemctl enable unattended-upgrades
  systemctl start unattended-upgrades

  # Configure fail2ban for SSH
  cat >/etc/fail2ban/jail.local <<'EOF'
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 5
bantime = 3600
findtime = 600
EOF

  systemctl enable fail2ban
  systemctl restart fail2ban

  # Configure UFW (Hetzner firewall is first layer, this is defense in depth)
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow 22/tcp comment 'SSH'
  ufw allow 80/tcp comment 'HTTP'
  ufw allow 443/tcp comment 'HTTPS'
  ufw --force enable

  # Disable root password login (key-only)
  sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
  sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
  systemctl reload sshd

  log "System hardening complete"
}

# -----------------------------------------------------------------------------
# Docker Installation
# -----------------------------------------------------------------------------
install_docker() {
  if command -v docker &>/dev/null; then
    log "Docker already installed, skipping..."
    return 0
  fi

  log "Installing Docker..."

  # Add Docker's official GPG key
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  # Add repository
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  # Configure Docker daemon
  mkdir -p /etc/docker
  cat >/etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF

  systemctl enable docker
  systemctl start docker

  log "Docker installed successfully"
}

# -----------------------------------------------------------------------------
# Generate Secrets
# -----------------------------------------------------------------------------
generate_secrets() {
  local env_file="${BASE_DIR}/.env"

  if [[ -f "${env_file}" ]]; then
    log "Secrets file exists, loading existing values..."
    # shellcheck source=/dev/null
    source "${env_file}"
    return 0
  fi

  log "Generating secrets..."

  mkdir -p "${BASE_DIR}"

  # Generate all passwords
  local pg_password; pg_password="$(generate_password)"
  local grafana_password; grafana_password="$(generate_password)"
  local minio_root_password; minio_root_password="$(generate_password)"
  local langfuse_secret; langfuse_secret="$(generate_password)"
  local langfuse_salt; langfuse_salt="$(generate_password)"
  local litellm_master_key; litellm_master_key="sk-$(generate_password)"
  local prometheus_password; prometheus_password="$(generate_password)"

  cat >"${env_file}" <<EOF
# Domain
DOMAIN=${DOMAIN}

# PostgreSQL
POSTGRES_PASSWORD=${pg_password}

# Grafana
GF_SECURITY_ADMIN_PASSWORD=${grafana_password}

# MinIO
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=${minio_root_password}

# Langfuse
NEXTAUTH_SECRET=${langfuse_secret}
SALT=${langfuse_salt}

# LiteLLM
LITELLM_MASTER_KEY=${litellm_master_key}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}

# Prometheus remote write auth
PROMETHEUS_REMOTE_USER=prometheus
PROMETHEUS_REMOTE_PASSWORD=${prometheus_password}

# Retention
LOKI_RETENTION_HOURS=$((LOKI_RETENTION_DAYS * 24))h
PROMETHEUS_RETENTION_DAYS=${PROMETHEUS_RETENTION_DAYS}d
EOF

  chmod 600 "${env_file}"

  log "Secrets generated and saved to ${env_file}"
  log "IMPORTANT: Save these credentials securely!"
  echo "=============================================="
  echo "Grafana admin password: ${grafana_password}"
  echo "MinIO root password: ${minio_root_password}"
  echo "LiteLLM master key: ${litellm_master_key}"
  echo "Prometheus remote password: ${prometheus_password}"
  echo "=============================================="
}

# -----------------------------------------------------------------------------
# Write Configuration Files
# -----------------------------------------------------------------------------
write_configs() {
  log "Writing configuration files..."

  # Source secrets
  # shellcheck source=/dev/null
  source "${BASE_DIR}/.env"

  # Create directories
  mkdir -p "${BASE_DIR}/caddy"
  mkdir -p "${BASE_DIR}/loki"
  mkdir -p "${BASE_DIR}/prometheus"
  mkdir -p "${BASE_DIR}/grafana/provisioning/datasources"
  mkdir -p "${BASE_DIR}/grafana/provisioning/dashboards"
  mkdir -p "${BASE_DIR}/litellm"
  mkdir -p "${BASE_DIR}/minio"
  mkdir -p "${BASE_DIR}/postgres-init"

  # -------------------------------------------------------------------------
  # Caddyfile
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/caddy/Caddyfile" <<EOF
# Global settings
{
  email admin@${DOMAIN}
}

# Grafana
grafana.${DOMAIN} {
  reverse_proxy grafana:3000
}

# Langfuse
langfuse.${DOMAIN} {
  reverse_proxy langfuse:3000
}

# Loki (push endpoint with basic auth)
loki.${DOMAIN} {
  @push {
    path /loki/api/v1/push
  }
  basic_auth @push {
    loki \$2a\$14\$placeholder
  }
  reverse_proxy loki:3100
}

# MLflow
mlflow.${DOMAIN} {
  reverse_proxy mlflow:5000
}

# LiteLLM
litellm.${DOMAIN} {
  reverse_proxy litellm:4000
}

# Prometheus (remote write with basic auth)
prometheus.${DOMAIN} {
  @write {
    path /api/v1/write
  }
  basic_auth @write {
    ${PROMETHEUS_REMOTE_USER} \$2a\$14\$placeholder
  }
  reverse_proxy prometheus:9090
}

# MinIO Console (optional, for debugging)
minio.${DOMAIN} {
  reverse_proxy minio:9001
}
EOF

  # Generate bcrypt hashes for Caddy basic auth
  local prom_hash; prom_hash=$(docker run --rm caddy:2 caddy hash-password --plaintext "${PROMETHEUS_REMOTE_PASSWORD}" 2>/dev/null || echo '$2a$14$placeholder')
  sed -i "s|\\\$2a\\\$14\\\$placeholder|${prom_hash}|g" "${BASE_DIR}/caddy/Caddyfile"

  # -------------------------------------------------------------------------
  # Loki config
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/loki/config.yaml" <<EOF
auth_enabled: false

server:
  http_listen_port: 3100
  log_level: info

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2024-01-01
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h

limits_config:
  retention_period: ${LOKI_RETENTION_HOURS}
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32
  max_streams_per_user: 10000
  max_line_size: 256kb

compactor:
  working_directory: /loki/compactor
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
  delete_request_store: filesystem

query_scheduler:
  max_outstanding_requests_per_tenant: 2048

querier:
  max_concurrent: 16
EOF

  # -------------------------------------------------------------------------
  # Prometheus config
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/prometheus/prometheus.yml" <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'hetzner-hub'

# Enable remote write receiver
# Prometheus will accept writes at /api/v1/write
# (Caddy handles auth before proxying)

scrape_configs:
  # Scrape self
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Scrape other services on this host
  - job_name: 'caddy'
    static_configs:
      - targets: ['caddy:2019']
    metrics_path: /metrics

  - job_name: 'loki'
    static_configs:
      - targets: ['loki:3100']
    metrics_path: /metrics

  - job_name: 'minio'
    static_configs:
      - targets: ['minio:9000']
    metrics_path: /minio/v2/metrics/cluster
EOF

  # -------------------------------------------------------------------------
  # Grafana datasources
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/grafana/provisioning/datasources/datasources.yaml" <<EOF
apiVersion: 1
deleteDatasources:
  - name: Prometheus
    orgId: 1
  - name: Loki
    orgId: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
  - name: Loki
    uid: loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false
EOF

  # -------------------------------------------------------------------------
  # LiteLLM config
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/litellm/config.yaml" <<EOF
model_list:
  # Claude models via Anthropic
  - model_name: claude-sonnet-4-20250514
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  # OpenAI models
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: o1
    litellm_params:
      model: openai/o1
      api_key: os.environ/OPENAI_API_KEY

  - model_name: o1-mini
    litellm_params:
      model: openai/o1-mini
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  drop_params: true
  set_verbose: false

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
EOF

  # -------------------------------------------------------------------------
  # PostgreSQL init script (create databases)
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/postgres-init/init.sql" <<'EOF'
-- Create databases for each service
CREATE DATABASE langfuse;
CREATE DATABASE mlflow;
CREATE DATABASE litellm;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE langfuse TO postgres;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;
GRANT ALL PRIVILEGES ON DATABASE litellm TO postgres;
EOF

  log "Configuration files written"
}

# -----------------------------------------------------------------------------
# Docker Compose
# -----------------------------------------------------------------------------
write_compose() {
  log "Writing docker-compose.yml..."

  # shellcheck source=/dev/null
  source "${BASE_DIR}/.env"

  cat >"${BASE_DIR}/docker-compose.yml" <<EOF
services:
  # =========================================================================
  # Reverse Proxy
  # =========================================================================
  caddy:
    image: caddy:2
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./caddy/Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy-data:/data
      - caddy-config:/config
    networks:
      - services
    depends_on:
      - grafana
      - langfuse
      - loki
      - mlflow
      - litellm
      - prometheus

  # =========================================================================
  # Database
  # =========================================================================
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./postgres-init:/docker-entrypoint-initdb.d:ro
    networks:
      - services
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # =========================================================================
  # Object Storage (for Langfuse)
  # =========================================================================
  minio:
    image: minio/minio:latest
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: \${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: \${MINIO_ROOT_PASSWORD}
    volumes:
      - minio-data:/data
    networks:
      - services
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MinIO bucket initialization
  minio-init:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set myminio http://minio:9000 \${MINIO_ROOT_USER} \${MINIO_ROOT_PASSWORD};
      mc mb --ignore-existing myminio/langfuse;
      mc ilm rule add --expire-days ${MINIO_RETENTION_DAYS} myminio/langfuse;
      exit 0;
      "
    networks:
      - services

  # =========================================================================
  # Langfuse (LLM Tracing)
  # =========================================================================
  langfuse:
    image: langfuse/langfuse:latest
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/langfuse
      NEXTAUTH_URL: https://langfuse.${DOMAIN}
      NEXTAUTH_SECRET: \${NEXTAUTH_SECRET}
      SALT: \${SALT}
      LANGFUSE_S3_MEDIA_UPLOAD_ENABLED: "true"
      LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_REGION: us-east-1
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: \${MINIO_ROOT_USER}
      LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: \${MINIO_ROOT_PASSWORD}
      LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: "true"
    networks:
      - services
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy

  # =========================================================================
  # Loki (Log Aggregation)
  # =========================================================================
  loki:
    image: grafana/loki:3.0.0
    restart: unless-stopped
    command: -config.file=/etc/loki/config.yaml
    volumes:
      - ./loki/config.yaml:/etc/loki/config.yaml:ro
      - loki-data:/loki
    networks:
      - services

  # =========================================================================
  # Prometheus (Metrics)
  # =========================================================================
  prometheus:
    image: prom/prometheus:v2.54.0
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=\${PROMETHEUS_RETENTION_DAYS}'
      - '--web.enable-remote-write-receiver'
      - '--web.enable-lifecycle'
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - services

  # =========================================================================
  # Grafana (Dashboards)
  # =========================================================================
  grafana:
    image: grafana/grafana-oss:11.2.0
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: \${GF_SECURITY_ADMIN_PASSWORD}
      GF_SERVER_ROOT_URL: https://grafana.${DOMAIN}
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    networks:
      - services

  # =========================================================================
  # MLflow (ML Experiment Tracking)
  # =========================================================================
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.17.0
    restart: unless-stopped
    command: >
      mlflow server
      --backend-store-uri postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/mlflow
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    networks:
      - services
    depends_on:
      postgres:
        condition: service_healthy

  # =========================================================================
  # LiteLLM (LLM Proxy)
  # =========================================================================
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    restart: unless-stopped
    command: --config /app/config.yaml --port 4000 --num_workers 4
    environment:
      LITELLM_MASTER_KEY: \${LITELLM_MASTER_KEY}
      ANTHROPIC_API_KEY: \${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: \${OPENAI_API_KEY}
      DATABASE_URL: postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/litellm
    volumes:
      - ./litellm/config.yaml:/app/config.yaml:ro
    networks:
      - services
    depends_on:
      postgres:
        condition: service_healthy

# =============================================================================
# Volumes
# =============================================================================
volumes:
  caddy-data:
  caddy-config:
  postgres-data:
  minio-data:
  loki-data:
  prometheus-data:
  grafana-data:
  mlflow-artifacts:

# =============================================================================
# Networks
# =============================================================================
networks:
  services:
    driver: bridge
EOF

  log "docker-compose.yml written"
}

# -----------------------------------------------------------------------------
# Start Services
# -----------------------------------------------------------------------------
start_services() {
  log "Starting services..."

  cd "${BASE_DIR}"
  docker compose pull
  docker compose up -d

  log "Waiting for services to be healthy..."
  sleep 10

  # Check service status
  docker compose ps

  log "Services started!"
  echo ""
  echo "=============================================="
  echo "Services are now running at:"
  echo "  - Grafana:    https://grafana.${DOMAIN}"
  echo "  - Langfuse:   https://langfuse.${DOMAIN}"
  echo "  - Loki:       https://loki.${DOMAIN}"
  echo "  - MLflow:     https://mlflow.${DOMAIN}"
  echo "  - LiteLLM:    https://litellm.${DOMAIN}"
  echo "  - Prometheus: https://prometheus.${DOMAIN}"
  echo "  - MinIO:      https://minio.${DOMAIN}"
  echo ""
  echo "Credentials saved to: ${BASE_DIR}/.env"
  echo "=============================================="
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
  require_root
  require_domain

  log "Starting Hetzner Cloud server setup for domain: ${DOMAIN}"

  harden_system
  install_docker
  generate_secrets
  write_configs
  write_compose
  start_services

  log "Setup complete!"
}

main "$@"
