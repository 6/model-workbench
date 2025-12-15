provider "hcloud" {
  token = var.hcloud_token
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

data "cloudflare_zone" "zone" {
  filter = { name = var.root_domain }
}

locals {
  cf_zone_id = data.cloudflare_zone.zone.zone_id
}

# Cloudflare edge IPs (so we can lock down Hetzner 80/443 to Cloudflare only)
data "cloudflare_ip_ranges" "cf" {}

resource "hcloud_ssh_key" "main" {
  name       = "${var.server_name}-key"
  public_key = var.ssh_public_key
}

resource "hcloud_firewall" "main" {
  name = "${var.server_name}-fw"

  # SSH only from your home/static IP CIDR(s)
  rule {
    direction   = "in"
    protocol    = "tcp"
    port        = "22"
    source_ips  = var.ssh_allowed_cidrs
    description = "SSH from allowed IPs only"
  }

  # HTTP/HTTPS only from Cloudflare (prevents bypassing Cloudflare)
  rule {
    direction   = "in"
    protocol    = "tcp"
    port        = "80"
    source_ips  = concat(data.cloudflare_ip_ranges.cf.ipv4_cidrs, data.cloudflare_ip_ranges.cf.ipv6_cidrs)
    description = "HTTP from Cloudflare only"
  }

  rule {
    direction   = "in"
    protocol    = "tcp"
    port        = "443"
    source_ips  = concat(data.cloudflare_ip_ranges.cf.ipv4_cidrs, data.cloudflare_ip_ranges.cf.ipv6_cidrs)
    description = "HTTPS from Cloudflare only"
  }

  # Optional: ping from your home only (or remove)
  rule {
    direction   = "in"
    protocol    = "icmp"
    source_ips  = var.ssh_allowed_cidrs
    description = "ICMP from allowed IPs only"
  }
}

resource "hcloud_server" "main" {
  name        = var.server_name
  image       = var.image
  server_type = var.server_type
  location    = var.location

  ssh_keys     = [hcloud_ssh_key.main.id]
  firewall_ids = [hcloud_firewall.main.id]

  backups = true

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }
}

resource "cloudflare_dns_record" "service_a" {
  for_each = toset(var.subdomains)

  zone_id = local.cf_zone_id
  name    = each.value
  type    = "A"
  content = hcloud_server.main.ipv4_address

  proxied = true
  ttl     = 1 # "Auto"
}

# Cloudflare SSL mode (recommended)
resource "cloudflare_zone_setting" "ssl_mode" {
  zone_id    = local.cf_zone_id
  setting_id = "ssl"
  value      = "strict"
}
