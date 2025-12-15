output "server_ipv4" {
  value = hcloud_server.main.ipv4_address
}

output "server_ipv6" {
  value = hcloud_server.main.ipv6_address
}

output "fqdn_services" {
  value = [for s in var.subdomains : "${s}.${var.root_domain}"]
}
