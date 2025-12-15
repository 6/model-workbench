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
  # Ubuntu 24.04 uses 'ssh' service name, older versions use 'sshd'
  systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true

  # Fix terminal compatibility for modern terminals (Ghostty, Kitty, Alacritty, etc.)
  if ! grep -q 'TERM=xterm-256color' /root/.bashrc 2>/dev/null; then
    echo 'export TERM=xterm-256color' >> /root/.bashrc
  fi

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
    log "Secrets file exists, checking for missing variables..."
    # shellcheck source=/dev/null
    source "${env_file}"

    # Add any missing variables (for upgrades)
    if [[ -z "${CLICKHOUSE_PASSWORD:-}" ]]; then
      log "Adding missing CLICKHOUSE_PASSWORD..."
      echo "" >> "${env_file}"
      echo "# ClickHouse (for Langfuse v3)" >> "${env_file}"
      echo "CLICKHOUSE_PASSWORD=$(generate_password)" >> "${env_file}"
      source "${env_file}"
    fi

    # Migrate from old separate auth vars to shared ADMIN_USER/PASSWORD
    if [[ -z "${ADMIN_USER:-}" ]]; then
      log "Adding shared ADMIN_USER/ADMIN_PASSWORD..."
      local admin_password; admin_password="$(generate_password)"
      echo "" >> "${env_file}"
      echo "# Shared basic auth (MLflow, Prometheus, Loki)" >> "${env_file}"
      echo "ADMIN_USER=admin" >> "${env_file}"
      echo "ADMIN_PASSWORD=${admin_password}" >> "${env_file}"
      source "${env_file}"
      echo "Shared basic auth (user: admin): ${admin_password}"
    fi
    return 0
  fi

  log "Generating secrets..."

  mkdir -p "${BASE_DIR}"

  # Generate all passwords
  local pg_password; pg_password="$(generate_password)"
  local grafana_password; grafana_password="$(generate_password)"
  local minio_root_password; minio_root_password="$(generate_password)"
  local clickhouse_password; clickhouse_password="$(generate_password)"
  local langfuse_secret; langfuse_secret="$(generate_password)"
  local langfuse_salt; langfuse_salt="$(generate_password)"
  local litellm_master_key; litellm_master_key="sk-$(generate_password)"
  local admin_password; admin_password="$(generate_password)"

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

# ClickHouse (for Langfuse v3)
CLICKHOUSE_PASSWORD=${clickhouse_password}

# Langfuse
NEXTAUTH_SECRET=${langfuse_secret}
SALT=${langfuse_salt}

# LiteLLM
LITELLM_MASTER_KEY=${litellm_master_key}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}

# Shared basic auth (MLflow, Prometheus, Loki)
ADMIN_USER=admin
ADMIN_PASSWORD=${admin_password}

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
  echo "Shared basic auth (user: admin): ${admin_password}"
  echo "  (used for MLflow, Prometheus, Loki)"
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

  # Validate required variables for basic auth
  if [[ -z "${ADMIN_USER:-}" ]] || [[ -z "${ADMIN_PASSWORD:-}" ]]; then
    error "ADMIN_USER or ADMIN_PASSWORD not set in ${BASE_DIR}/.env"
  fi

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

# Loki (full site basic auth)
loki.${DOMAIN} {
  basic_auth {
    ${ADMIN_USER} \$2a\$14\$admin_placeholder
  }
  reverse_proxy loki:3100
}

# MLflow (full site basic auth)
mlflow.${DOMAIN} {
  basic_auth {
    ${ADMIN_USER} \$2a\$14\$admin_placeholder
  }
  reverse_proxy mlflow:5000
}

# LiteLLM (API key auth via LITELLM_MASTER_KEY)
litellm.${DOMAIN} {
  reverse_proxy litellm:4000
}

# Prometheus (full site basic auth)
prometheus.${DOMAIN} {
  basic_auth {
    ${ADMIN_USER} \$2a\$14\$admin_placeholder
  }
  reverse_proxy prometheus:9090
}

# MinIO Console (optional, for debugging)
minio.${DOMAIN} {
  reverse_proxy minio:9001
}
EOF

  # Generate bcrypt hash for shared basic auth (used by MLflow, Prometheus, Loki)
  local admin_hash; admin_hash=$(docker run --rm caddy:2 caddy hash-password --plaintext "${ADMIN_PASSWORD}" 2>/dev/null || echo '$2a$14$admin_placeholder')
  sed -i "s|\\\$2a\\\$14\\\$admin_placeholder|${admin_hash}|g" "${BASE_DIR}/caddy/Caddyfile"

  # -------------------------------------------------------------------------
  # Loki config
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/loki/config.yaml" <<EOF
auth_enabled: false

analytics:
  reporting_enabled: false

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

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
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
  # Grafana dashboard provisioning
  # -------------------------------------------------------------------------
  cat >"${BASE_DIR}/grafana/provisioning/dashboards/dashboards.yaml" <<EOF
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

  # Hetzner Server Dashboard
  cat >"${BASE_DIR}/grafana/provisioning/dashboards/hetzner.json" <<'DASHBOARD_EOF'
{
  "annotations": {"list": []},
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "palette-classic"},
          "mappings": [],
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 70}, {"color": "red", "value": 90}]},
          "unit": "percent",
          "min": 0,
          "max": 100
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
      "id": 1,
      "options": {"orientation": "auto", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "showThresholdLabels": false, "showThresholdMarkers": true},
      "pluginVersion": "11.2.0",
      "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)", "refId": "A"}],
      "title": "CPU Usage",
      "type": "gauge"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "palette-classic"},
          "mappings": [],
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 70}, {"color": "red", "value": 90}]},
          "unit": "percent",
          "min": 0,
          "max": 100
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
      "id": 2,
      "options": {"orientation": "auto", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "showThresholdLabels": false, "showThresholdMarkers": true},
      "pluginVersion": "11.2.0",
      "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "100 * (1 - ((node_memory_MemAvailable_bytes or node_memory_Buffers_bytes + node_memory_Cached_bytes + node_memory_MemFree_bytes) / node_memory_MemTotal_bytes))", "refId": "A"}],
      "title": "Memory Usage",
      "type": "gauge"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "palette-classic"},
          "mappings": [],
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}, {"color": "yellow", "value": 70}, {"color": "red", "value": 90}]},
          "unit": "percent",
          "min": 0,
          "max": 100
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
      "id": 3,
      "options": {"orientation": "auto", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "showThresholdLabels": false, "showThresholdMarkers": true},
      "pluginVersion": "11.2.0",
      "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "100 - ((node_filesystem_avail_bytes{mountpoint=\"/\",fstype!=\"rootfs\"} / node_filesystem_size_bytes{mountpoint=\"/\",fstype!=\"rootfs\"}) * 100)", "refId": "A"}],
      "title": "Disk Usage (/)",
      "type": "gauge"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "palette-classic"},
          "mappings": [],
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}]},
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
      "id": 4,
      "options": {"orientation": "auto", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false}, "showThresholdLabels": false, "showThresholdMarkers": true},
      "pluginVersion": "11.2.0",
      "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "node_load1", "refId": "A"}],
      "title": "Load (1m)",
      "type": "gauge"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "palette-classic"},
          "custom": {"axisBorderShow": false, "axisCenteredZero": false, "axisColorMode": "text", "axisLabel": "", "axisPlacement": "auto", "barAlignment": 0, "drawStyle": "line", "fillOpacity": 10, "gradientMode": "none", "hideFrom": {"legend": false, "tooltip": false, "viz": false}, "insertNulls": false, "lineInterpolation": "linear", "lineWidth": 1, "pointSize": 5, "scaleDistribution": {"type": "linear"}, "showPoints": "never", "spanNulls": false, "stacking": {"group": "A", "mode": "none"}, "thresholdsStyle": {"mode": "off"}},
          "mappings": [],
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}]},
          "unit": "Bps"
        },
        "overrides": [{"matcher": {"id": "byName", "options": "receive"}, "properties": [{"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}]}, {"matcher": {"id": "byName", "options": "transmit"}, "properties": [{"id": "color", "value": {"fixedColor": "blue", "mode": "fixed"}}]}]
      },
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
      "id": 5,
      "options": {"legend": {"calcs": [], "displayMode": "list", "placement": "bottom", "showLegend": true}, "tooltip": {"mode": "multi", "sort": "none"}},
      "pluginVersion": "11.2.0",
      "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "irate(node_network_receive_bytes_total{device=~\"eth.*|ens.*\"}[5m])", "legendFormat": "receive", "refId": "A"}, {"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "irate(node_network_transmit_bytes_total{device=~\"eth.*|ens.*\"}[5m])", "legendFormat": "transmit", "refId": "B"}],
      "title": "Network I/O",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {
        "defaults": {
          "color": {"mode": "palette-classic"},
          "custom": {"axisBorderShow": false, "axisCenteredZero": false, "axisColorMode": "text", "axisLabel": "", "axisPlacement": "auto", "barAlignment": 0, "drawStyle": "line", "fillOpacity": 10, "gradientMode": "none", "hideFrom": {"legend": false, "tooltip": false, "viz": false}, "insertNulls": false, "lineInterpolation": "linear", "lineWidth": 1, "pointSize": 5, "scaleDistribution": {"type": "linear"}, "showPoints": "never", "spanNulls": false, "stacking": {"group": "A", "mode": "none"}, "thresholdsStyle": {"mode": "off"}},
          "mappings": [],
          "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": null}]},
          "unit": "percent"
        },
        "overrides": []
      },
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
      "id": 6,
      "options": {"legend": {"calcs": [], "displayMode": "list", "placement": "bottom", "showLegend": true}, "tooltip": {"mode": "multi", "sort": "none"}},
      "pluginVersion": "11.2.0",
      "targets": [{"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)", "legendFormat": "CPU", "refId": "A"}, {"datasource": {"type": "prometheus", "uid": "prometheus"}, "expr": "100 * (1 - ((node_memory_MemAvailable_bytes or node_memory_Buffers_bytes + node_memory_Cached_bytes + node_memory_MemFree_bytes) / node_memory_MemTotal_bytes))", "legendFormat": "Memory", "refId": "B"}],
      "title": "CPU & Memory Over Time",
      "type": "timeseries"
    }
  ],
  "schemaVersion": 39,
  "tags": ["node", "system"],
  "templating": {"list": []},
  "time": {"from": "now-1h", "to": "now"},
  "timepicker": {},
  "timezone": "browser",
  "title": "Hetzner Server",
  "uid": "hetzner-server",
  "version": 1
}
DASHBOARD_EOF

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
    # Note: No depends_on - Caddy should start regardless of backend health

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
      mc ilm rule add --expire-days ${MINIO_RETENTION_DAYS} myminio/langfuse || true;
      exit 0;
      "
    networks:
      - services

  # =========================================================================
  # ClickHouse (required for Langfuse v3)
  # =========================================================================
  clickhouse:
    image: clickhouse/clickhouse-server:24.3
    restart: unless-stopped
    environment:
      CLICKHOUSE_DB: langfuse
      CLICKHOUSE_USER: langfuse
      CLICKHOUSE_PASSWORD: \${CLICKHOUSE_PASSWORD}
    volumes:
      - clickhouse-data:/var/lib/clickhouse
    networks:
      - services
    healthcheck:
      test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  # =========================================================================
  # Redis (required for Langfuse v3 job queue)
  # =========================================================================
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - services
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # =========================================================================
  # Langfuse (LLM Tracing)
  # =========================================================================
  langfuse:
    image: langfuse/langfuse:latest
    restart: unless-stopped
    environment:
      DATABASE_URL: postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/langfuse
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000/langfuse
      CLICKHOUSE_USER: langfuse
      CLICKHOUSE_PASSWORD: \${CLICKHOUSE_PASSWORD}
      CLICKHOUSE_CLUSTER_ENABLED: "false"
      REDIS_CONNECTION_STRING: redis://redis:6379
      NEXTAUTH_URL: https://langfuse.${DOMAIN}
      NEXTAUTH_SECRET: \${NEXTAUTH_SECRET}
      SALT: \${SALT}
      TELEMETRY_ENABLED: "false"
      LANGFUSE_S3_BATCH_EXPORT_ENABLED: "false"
      LANGFUSE_S3_EVENT_UPLOAD_ENABLED: "true"
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_REGION: us-east-1
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: \${MINIO_ROOT_USER}
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: \${MINIO_ROOT_PASSWORD}
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
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
      clickhouse:
        condition: service_healthy
      redis:
        condition: service_healthy

  # =========================================================================
  # Langfuse Worker (processes events into ClickHouse)
  # =========================================================================
  langfuse-worker:
    image: langfuse/langfuse:latest
    restart: unless-stopped
    command: ["node", "packages/worker/dist/index.js"]
    environment:
      DATABASE_URL: postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/langfuse
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000/langfuse
      CLICKHOUSE_USER: langfuse
      CLICKHOUSE_PASSWORD: \${CLICKHOUSE_PASSWORD}
      CLICKHOUSE_CLUSTER_ENABLED: "false"
      REDIS_CONNECTION_STRING: redis://redis:6379
      SALT: \${SALT}
      TELEMETRY_ENABLED: "false"
      LANGFUSE_S3_EVENT_UPLOAD_ENABLED: "true"
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_REGION: us-east-1
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: \${MINIO_ROOT_USER}
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: \${MINIO_ROOT_PASSWORD}
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
    networks:
      - services
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      clickhouse:
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
  # Node Exporter (System Metrics)
  # =========================================================================
  node-exporter:
    image: prom/node-exporter:v1.8.2
    restart: unless-stopped
    command:
      - '--path.rootfs=/host'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    volumes:
      - /:/host:ro,rslave
    networks:
      - services
    pid: host

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
      GF_ANALYTICS_REPORTING_ENABLED: "false"
      GF_ANALYTICS_CHECK_FOR_UPDATES: "false"
      GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES: "false"
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
    environment:
      MLFLOW_DISABLE_TELEMETRY: "true"
    entrypoint: /bin/sh
    command:
      - -c
      - |
        pip install --quiet psycopg2-binary && \
        mlflow server \
          --backend-store-uri postgresql://postgres:\${POSTGRES_PASSWORD}@postgres:5432/mlflow \
          --default-artifact-root /mlflow/artifacts \
          --host 0.0.0.0 \
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
  clickhouse-data:
  redis-data:
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

  # Reload Caddy to ensure config changes are applied
  log "Reloading Caddy configuration..."
  docker compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile || true

  # Check service status
  docker compose ps

  # Create systemd service for auto-start on boot
  log "Creating systemd service for auto-start on boot..."
  cat >/etc/systemd/system/docker-compose-services.service <<EOF
[Unit]
Description=Docker Compose Services
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${BASE_DIR}
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable docker-compose-services.service

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
