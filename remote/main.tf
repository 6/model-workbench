provider "hcloud" {
  token = var.hcloud_token
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

data "cloudflare_zone" "zone" {
  filter = {
    name = var.root_domain
  }
}

locals {
  cf_zone_id = data.cloudflare_zone.zone.zone_id
}

resource "hcloud_ssh_key" "main" {
  name       = "${var.server_name}-key"
  public_key = var.ssh_public_key
}

resource "hcloud_firewall" "main" {
  name = "${var.server_name}-fw"

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = var.ssh_allowed_cidrs
    description = "SSH from allowed IPs only"
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
    description = "HTTP"
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = ["0.0.0.0/0", "::/0"]
    description = "HTTPS"
  }

  # Optional: ping from anywhere (handy for debugging)
  rule {
    direction  = "in"
    protocol   = "icmp"
    source_ips = ["0.0.0.0/0", "::/0"]
    description = "ICMP"
  }
}

resource "hcloud_server" "main" {
  name        = var.server_name
  image       = var.image
  server_type = var.server_type
  location    = var.location

  ssh_keys    = [hcloud_ssh_key.main.id]
  firewall_ids = [hcloud_firewall.main.id]

  # Hetzner “Backups” feature (keeps multiple backup slots; Hetzner describes 7 slots) :contentReference[oaicite:2]{index=2}
  backups = true

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }
}

locals {
  services = toset(var.subdomains)
}

resource "cloudflare_dns_record" "service_a" {
  for_each = toset(var.subdomains)

  zone_id  = local.cf_zone_id
  name     = each.value           # "grafana", "langfuse", etc (relative to zone)
  type     = "A"
  content  = hcloud_server.main.ipv4_address
  proxied  = var.cloudflare_proxied
  ttl      = 1
}

# Set SSL/TLS mode to Full (strict)
resource "cloudflare_zone_setting" "ssl_mode" {
  zone_id    = local.cf_zone_id
  setting_id = "ssl"
  value      = "strict"
}
