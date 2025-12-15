# Remote Observability Hub

Hetzner Cloud setup for centralized observability and LLM proxy services.

## Services

| Service | URL | Auth |
|---------|-----|------|
| **Grafana** | `https://grafana.{domain}` | Master credentials |
| **Langfuse** | `https://langfuse.{domain}` | Master credentials (auto-created) |
| **Loki** | `https://loki.{domain}` | Master credentials |
| **MLflow** | `https://mlflow.{domain}` | Master credentials |
| **LiteLLM** | `https://litellm.{domain}` | API key (`LITELLM_MASTER_KEY`) |
| **Prometheus** | `https://prometheus.{domain}` | Master credentials |
| **MinIO** | `https://minio.{domain}` | Master credentials |

**Master credentials**: Single `ADMIN_USER` / `ADMIN_PASSWORD` from `/opt/services/.env` works for all services.

## Prerequisites

- Terraform 1.6+
- Domain with Cloudflare DNS
- Hetzner Cloud account

## Deployment

### 1. Configure Environment

Copy `.envrc.example` to `.envrc` in the repo root and fill in values:

```bash
cp ../.envrc.example ../.envrc
# Edit with your tokens, domain, SSH key, etc.
direnv allow
```

### 2. Deploy Infrastructure (Terraform)

```bash
# From repo root
make tf-init
make tf-plan
make tf-apply
```

This creates:
- Hetzner Cloud server (cx53: 32GB RAM, 16 vCPU, 320GB disk)
- Cloudflare DNS records for all subdomains
- Firewall rules (SSH from allowed IPs, HTTP/HTTPS from Cloudflare only)

### 3. Bootstrap Server

SSH to the server and run the setup script:

```bash
# Get server IP
cd remote && terraform output server_ipv4

# SSH and run setup
ssh root@<server-ip>

# On the server - option A: run directly
curl -fsSL https://raw.githubusercontent.com/<your-repo>/main/remote/setup.sh | DOMAIN=example.com bash

# On the server - option B: copy and run
# (first scp the setup.sh file)
DOMAIN=example.com \
ANTHROPIC_API_KEY=sk-ant-... \
OPENAI_API_KEY=sk-... \
./setup.sh
```

The script will:
- Harden the system (unattended-upgrades, fail2ban, UFW)
- Install Docker
- Generate a single master password for all services
- Create systemd service for auto-start on boot
- Start all containers
- Print credentials to save

### 4. Save Credentials

After setup completes, save the generated credentials:

```bash
# On the server - credentials printed at end of setup:
# MASTER PASSWORD (user: admin): xxxxx
#   → Grafana, MinIO, MLflow, Prometheus, Loki, Langfuse
# LiteLLM API key: sk-xxxxx

# Or view later:
cat /opt/services/.env | grep -E "^ADMIN_PASSWORD|^LITELLM_MASTER_KEY"
```

### 5. Configure Local Prometheus (Optional)

To push metrics from your local GPU server to the remote Prometheus:

1. Copy `local-prometheus-remote-write.yaml` config snippet
2. Add it to your local `prometheus.yml`
3. Set environment variables:
   ```bash
   export REMOTE_PROMETHEUS_URL="https://prometheus.example.com"
   export REMOTE_ADMIN_USER="admin"
   export REMOTE_ADMIN_PASSWORD="<ADMIN_PASSWORD from server>"
   ```
4. Restart Prometheus

### 6. Configure Local Loki Push (Optional)

To push logs from your local server to the remote Loki:

```bash
# Using promtail or any Loki client:
export LOKI_URL="https://loki.example.com"
export LOKI_USER="admin"
export LOKI_PASSWORD="<ADMIN_PASSWORD from server>"

# Example promtail config:
# clients:
#   - url: https://loki.example.com/loki/api/v1/push
#     basic_auth:
#       username: admin
#       password: ${LOKI_PASSWORD}
```

## Storage & Retention

| Component | Allocation | Retention |
|-----------|-----------|-----------|
| Loki logs | ~100GB | 14 days |
| Langfuse traces (MinIO) | ~80GB | 30 days |
| Prometheus metrics | ~40GB | 14 days |
| PostgreSQL | ~20GB | N/A |
| MLflow artifacts | ~40GB | Manual |
| OS/Docker/Buffer | ~40GB | N/A |

Total: ~320GB (matches server disk)

## Managing Services

```bash
# SSH to server
ssh root@<server-ip>
cd /opt/services

# View logs
docker compose logs -f langfuse
docker compose logs -f litellm

# Restart a service
docker compose restart langfuse

# Update all images
docker compose pull
docker compose up -d

# Check disk usage
df -h
du -sh /var/lib/docker/volumes/*
```

## LiteLLM Usage

```bash
# Get your master key from /opt/services/.env on server

# Test the proxy
curl -X POST https://litellm.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Available models:
- `claude-sonnet-4-20250514`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`

### Updating API Keys

To add or update API keys after initial setup:

```bash
ssh root@<server-ip>
cd /opt/services

# Edit .env file
nano .env  # or vim

# Update these lines:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Restart LiteLLM to pick up changes
docker compose restart litellm

# Verify
docker compose logs litellm | head -20
```

## MLflow Usage

MLflow uses master credentials (`ADMIN_USER` / `ADMIN_PASSWORD`):

```bash
# Access via browser with credentials, or via CLI:
export MLFLOW_TRACKING_URI="https://admin:<ADMIN_PASSWORD>@mlflow.example.com"

# Or configure in Python
import mlflow
mlflow.set_tracking_uri("https://mlflow.example.com")
# Auth via environment: MLFLOW_TRACKING_USERNAME=admin, MLFLOW_TRACKING_PASSWORD=<ADMIN_PASSWORD>
```

## Files

```
remote/
├── main.tf                 # Terraform: Hetzner + Cloudflare
├── variables.tf            # Terraform variables
├── outputs.tf              # Terraform outputs
├── versions.tf             # Provider versions
├── setup.sh                # Server bootstrap (generates all configs)
├── local-prometheus-remote-write.yaml  # Config snippet for local Prometheus
└── README.md
```

Note: `setup.sh` generates all Docker Compose and service configs on the server at `/opt/services/`.

## Troubleshooting

### Services not starting

```bash
docker compose ps
docker compose logs <service-name>
```

### Caddy TLS issues

Cloudflare SSL/TLS mode must be "Full (strict)" for Caddy auto-TLS to work.
Check Cloudflare dashboard → SSL/TLS → Overview.

### Prometheus remote write failing

1. Check credentials in local prometheus config
2. Verify Caddy is running: `docker compose logs caddy`
3. Test auth: `curl -u admin:<ADMIN_PASSWORD> https://prometheus.example.com/api/v1/query?query=up`

### Disk full

```bash
# Check what's using space
du -sh /var/lib/docker/volumes/*

# Force Loki compaction
docker compose restart loki

# Prune unused Docker data
docker system prune -a
```
