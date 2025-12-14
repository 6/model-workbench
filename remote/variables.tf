variable "hcloud_token" {
  type      = string
  sensitive = true
}

variable "cloudflare_api_token" {
  type      = string
  sensitive = true
}

variable "root_domain" {
  type        = string
  description = "Your Cloudflare zone, e.g. example.com"
}

variable "server_name" {
  type        = string
  description = "Hetzner server name"
}

variable "location" {
  type        = string
  description = "Hetzner location (fsn1 = Falkenstein)"
}

variable "server_type" {
  type        = string
  description = "Hetzner server type"
}

variable "image" {
  type        = string
  description = "Hetzner image"
}

variable "ssh_public_key" {
  type        = string
  description = "Your public key contents (ed25519 pubkey string)"
  sensitive   = true
}

variable "ssh_allowed_cidrs" {
  type        = list(string)
  description = "CIDRs allowed to SSH (port 22). If your IP changes, update this and re-apply."
}

variable "subdomains" {
  type        = list(string)
  default     = ["grafana", "loki", "langfuse", "litellm", "mlflow"]
  description = "Subdomains to create A/AAAA records for"
}
