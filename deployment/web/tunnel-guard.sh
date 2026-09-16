#!/usr/bin/env bash
# Limit the tunnel identity's new loopback connections to the public API port.
# Only this dedicated nftables table is replaced; other host rules are preserved.
set -euo pipefail
test "$(id -u)" -eq 0
tunnel_uid="$(id -u opendpd-tunnel)"
tunnel_delete=''
if nft list table inet opendpd_tunnel >/dev/null 2>&1; then
  tunnel_delete='delete table inet opendpd_tunnel'
fi
nft -f - <<EOF
$tunnel_delete
table inet opendpd_tunnel {
  chain local_destination {
    ip daddr 127.0.0.1 tcp dport 18765 accept
    reject
  }
  chain output {
    type filter hook output priority 0; policy accept;
    meta skuid $tunnel_uid ct state new ip daddr 127.0.0.0/8 jump local_destination
    meta skuid $tunnel_uid ct state new ip6 daddr ::1 jump local_destination
  }
}
EOF
