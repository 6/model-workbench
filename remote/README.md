# Remote Observability Hub

Hetzner Cloud setup for centralized observability and LLM proxy services.

## Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Grafana** | `https://grafana.{domain}` | Dashboards and visualization |
| **Langfuse** | `https://langfuse.{domain}` | LLM tracing and observability |
| **Loki** | `https://loki.{domain}` | Log aggregation |
| **MLflow** | `https://mlflow.{domain}` | ML experiment tracking |
| **LiteLLM** | `https://litellm.{domain}` | LLM API proxy (Anthropic/OpenAI) |
| **Prometheus** | `https://prometheus.{domain}` | Metrics storage (remote write) |
| **MinIO** | `https://minio.{domain}` | S3-compatible storage (for Langfuse) |

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
- Generate random passwords for all services
- Start all containers
- Print credentials to save

### 4. Save Credentials

After setup completes, save the generated credentials:

```bash
# On the server
cat /opt/services/.env

# Copy PROMETHEUS_REMOTE_PASSWORD to your local .envrc
```

### 5. Configure Local Prometheus (Optional)

To push metrics from your local GPU server to the remote Prometheus:

1. Copy `local-prometheus-remote-write.yaml` config snippet
2. Add it to your local `prometheus.yml`
3. Set environment variables:
   ```bash
   export REMOTE_PROMETHEUS_URL="https://prometheus.example.com"
   export REMOTE_PROMETHEUS_USER="prometheus"
   export REMOTE_PROMETHEUS_PASSWORD="<from-server>"
   ```
4. Restart Prometheus

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

## Files

```
remote/
├── main.tf                 # Terraform: Hetzner + Cloudflare
├── variables.tf            # Terraform variables
├── outputs.tf              # Terraform outputs
├── versions.tf             # Provider versions
├── setup.sh                # Server bootstrap script
├── docker-compose.yml      # All services
├── .env.example            # Environment template
├── local-prometheus-remote-write.yaml  # Config for local server
└── configs/
    ├── Caddyfile           # Reverse proxy routes
    ├── loki-config.yaml    # Loki with 14-day retention
    ├── prometheus.yml      # Prometheus config
    ├── litellm-config.yaml # LLM routing
    ├── postgres-init.sql   # Database init
    └── grafana/
        └── provisioning/
            └── datasources/
                └── datasources.yaml
```

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
3. Test auth: `curl -u prometheus:<password> https://prometheus.example.com/api/v1/query?query=up`

### Disk full

```bash
# Check what's using space
du -sh /var/lib/docker/volumes/*

# Force Loki compaction
docker compose restart loki

# Prune unused Docker data
docker system prune -a
```
