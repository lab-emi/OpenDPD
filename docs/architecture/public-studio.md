# GitHub Pages Studio and isolated local compute

The public Studio is a static build at `https://opendpd.com/studio/`. It calls
`https://api.opendpd.com` through a named Cloudflare Tunnel. Dataset analysis,
preprocessing, training, inference and result generation execute inside the
dedicated local QEMU VM. GitHub Pages serves JavaScript, styles and images only.
This deployment replaces the earlier [cloud pull proposal](cloud-pull-deployment-plan.md).

```text
Browser ──HTTPS──> GitHub Pages: opendpd.com/studio/
   └──────HTTPS──> Cloudflare: api.opendpd.com/api/v1/…
                         │ named tunnel, outbound connection from host
                  cloudflared (opendpd-tunnel user)
                         │ 127.0.0.1:18765
                  QEMU (opendpd-sandbox user, restrict=on)
                         │ guest :8765
                  Public API boundary (worker user)
                         ├── random session A / private database / supervisor
                         └── random session B / private database / supervisor
                                   └── one shared compute slot
```

## Sessions and deletion

**Use random session directories, never an IP address as a workspace identity.**
People behind a university, company, mobile network or VPN can share an IP.
An IP also changes and is not an authentication credential. Two visitors using
the same IP receive independent workspaces and cannot list or download each
other's files. A browser tab stores its random bearer token in `sessionStorage`;
tokens are sent only in the Authorization header, never in links or query strings.
The API retains only token hashes in memory. A token is not bound to an IP.

```text
/run/opendpd-web/                    # private 2 GiB tmpfs, no swap
  sessions/<random 48-hex id>/       # permissions 0700
    lease.json                      # fixed expiry, no raw IP or token
    workspace/                      # inputs, preprocessing, runs, exports, cache
  service-tmp/                      # shared library temporary files
```

All sessions expire at **23:55 UTC**. New sessions pause until 00:00 UTC.
Every 15 seconds the API stops expired workspaces and deletes their files.
Independently, a systemd timer at **23:59 UTC** restarts the entire API cgroup:
it terminates remaining requests and workers, then discards its private tmpfs.
The stop timeout is 20 seconds with SIGKILL as the final fallback. This leaves
margin before any session data reaches 24 hours. Sessions do not get a fresh
24 hours when they are accessed or when new files are generated. Late visitors
therefore have a shorter session; the UI displays their exact expiry.

Restarting the API or VM also discards every session. There is no resume,
backup, persistent user database or recoverable job history in this deployment.
Swap and core dumps are disabled. Do not introduce VM memory snapshots,
hibernation, workspace backups or request-body logging: those would invalidate
the retention design. If the host is suspended across the deadline, the
persistent timer runs on resume; deletion cannot execute while the host is
asleep. Operate this host continuously and keep its clock synchronized.
User-downloaded files and Cloudflare/GitHub platform logs are outside this
server-side file deletion policy. Raw client IPs are not stored by the app;
Cloudflare necessarily processes them at the edge.

Uploads of datasets, checkpoints, packages and code, local-path imports, shell
execution, RF control and executable deployment exports are unavailable. Only
reviewed built-in datasets and supported neural models are published. Any
future upload feature must write into the same session tree and add file-format,
decompression, size and model-loading validation before its route is allowed.

## Security and capacity

The public entrypoint is `python -m opendpd.web`, **not** `opendpd gui`.
An explicit route allowlist sits ahead of separate local-app instances; new
desktop routes are not automatically public. Each app has its own workspace,
SQLite database and supervisor. Read, download, event polling, cancel and retry
requests all resolve within the authenticated workspace.

The host listens for the API and administrative SSH only on loopback, at ports
18765 and 22224. QEMU uses `restrict=on`, no bridge and no guest-initiated Internet
or LAN connections. The guest service and its workers cannot modify source or
the runtime installation. Resource controls apply to the entire service cgroup;
workers inherit a small environment without tunnel credentials.

Cloudflared rewrites the HTTP Host to a private, random origin credential.
The API requires that Host, a trusted direct peer, HTTPS forwarding metadata,
one valid `CF-Connecting-IP`, and exactly the configured browser Origin.
Uvicorn proxy-header handling is disabled, and `X-Forwarded-For` is ignored.
The public hostname alone cannot reach the origin through a forged local
request. Keep the private Host and tunnel credential out of Git, browser builds
and logs. Trusted host administrators remain inside the trust boundary.

CORS uses one explicit origin and no cookies. API responses are `no-store` with
security headers. The frontend has a restrictive CSP; authenticated downloads
reject redirects and foreign origins. Add `frame-ancestors 'none'` as a
Cloudflare response header for `/studio/` because a meta CSP cannot enforce it.

| Limit | Default |
|---|---|
| Live sessions | 16 globally; 8 created per IP per UTC day |
| HTTP requests | 120/minute per IP, including preflight; 8 concurrent globally, 3/session |
| JSON request body | 64 KiB, 10-second receive deadline; no multipart |
| Jobs | 8/session, 12/IP/day, 60/day globally; 2 pending/session |
| Concurrent training/inference | 1 globally across every session |
| Job runtime | 30 minutes, followed by cancellation and forced termination |
| Public model/training parameters | bounded layers, widths, batches, frames, epochs and threads |
| Storage | 256 MiB/session checked every 15 seconds; **2 GiB hard limit globally** |
| Files/processes | 64 MiB/file, 256 tasks, 4 CPU equivalents, 6 GiB API cgroup RAM |

IPv6 addresses share a /64 rate-limit bucket. IP quotas use an ephemeral HMAC
key and reset on service restart. They discourage abuse but do not identify
people or stop distributed clients. Global concurrency, memory, disk and job
budgets bound the effect of IP rotation. There is no Turnstile implementation;
do not describe this demo as bot-proof. Cloudflare edge controls and monitoring
remain useful for floods that exceed the local admission layer.

The installed baseline is a CPU VM with 4 vCPUs and 8 GiB RAM. GPU passthrough
is a separate deployment change; the existing GPU validation VM is unchanged.
The public API can support an intentionally configured CUDA device, but the
CPU deployment does not advertise an available GPU.

## Deployment

Use the versioned units and scripts in
[`deployment/web`](https://github.com/lab-emi/OpenDPD/tree/OpenDPD-Studio/deployment/web).

1. Preserve the existing DNS records before changing nameservers. For this
   domain the initial scan found four GitHub Pages A records, four AAAA records
   and `www → lab-emi.github.io`. Squarespace additionally contains three
   GitHub verification TXT records at `_github-pages-challenge-gaochangw`,
   `_github-pages-challenge-lab-emi` and
   `_github-pages-challenge-gaochangw.opendpd.com` (the last name is relative
   to the zone, so its full name ends in `.opendpd.com.opendpd.com`). Preserve
   these records too; an automatic apex DNS scan misses them. Keep registration at
   Squarespace; change DNS hosting only. Cloudflare assigned
   `ali.ns.cloudflare.com` and `scott.ns.cloudflare.com`. Check authoritative
   NS and DS records before and after switching. Never remove an active DNSSEC
   chain without coordinating the registrar change.
2. The existing custom domain belongs to `lab-emi/opendpd-site`. Its Pages
   workflow builds the landing page, checks out a reviewed commit of this
   repository, builds `frontend/` with `VITE_STUDIO_MODE=web`,
   `VITE_API_ORIGIN=https://api.opendpd.com` and relative base URLs, then adds
   that build at `/studio/` in **one** Pages artifact. This preserves the
   existing homepage and makes the frontend run at the exact allowed Origin.
   Set that workflow's `STUDIO_REF` variable or its `studio_ref` dispatch input
   to deploy a later reviewed OpenDPD revision. Updating this repository alone
   does not update the production frontend. This repository's `Docs` workflow
   validates the web build and publishes documentation with a redirect from
   its `/studio/` to the canonical app. No API secrets belong in either
   repository's Actions variables or frontend build variables.
3. On the existing sandbox host, stop the CPU validation VM before running
   `sudo bash deployment/web/provision-vm.sh`. It creates a separate frozen
   base, writable web overlay and SSH public-key seed. It refuses to overwrite
   an existing web disk and does not switch GPUs.
4. Prepare the Python runtime **offline** for the guest's OS, architecture and
   Python version, with the project's core and `gui` dependencies plus `uv`,
   `setuptools`, `setuptools_scm` and `wheel`. Copy source to `/opt/opendpd` and
   the runtime to `/opt/opendpd-venv`. Existing validation-container runtimes
   can be reused, but install this source revision into the venv: a different
   working directory alone will not update training subprocess imports.
5. Create root-only `/etc/opendpd-web.env` inside the guest:

   ```dotenv
   OPENDPD_WEB_ORIGIN=https://opendpd.com
   OPENDPD_WEB_API_HOST=api.opendpd.com
   OPENDPD_WEB_TUNNEL_HOST=REPLACE_WITH_RANDOM_48_HEX.internal
   ```

   Generate the private value with `secrets.token_hex(24) + '.internal'` and
   put the identical value in cloudflared's `httpHostHeader`. Do not use the
   placeholder. Run `sudo bash /opt/opendpd/deployment/web/install-guest.sh`.
   The provided units require worker UID/GID 1001 and systemd with tmpfs and
   cgroup resource controls.
6. Create a **named**, dashboard-managed Cloudflare Tunnel for
   `api.opendpd.com`. The installed production tunnel is `opendpd-studio`.
   Configure one published application with hostname `api.opendpd.com`, path
   `^/api/v1/.*$`, service `http://127.0.0.1:18765`, and the private HTTP Host
   from step 5. Retain the default 404 for unmatched requests. Save its token
   at host `/etc/opendpd-web/tunnel.token` (0600), owned by the dedicated
   `opendpd-tunnel` account. Install `cloudflared-opendpd-remote.service`;
   the token is read from that file and is absent from command-line arguments.
   An account-wide management certificate is not needed by the service.
   Publish no SSH, TCP, private-network or catch-all service routes.
   Never use a Quick Tunnel for this service.
7. Enable the connector and `opendpd-web-vm.service`. Verify that Cloudflare
   reports it healthy and that local `127.0.0.1:20242/ready` has active
   connections. The alternative `cloudflared.yml.example` and
   `cloudflared-opendpd.service` are for a **locally managed** tunnel with a
   JSON credential; do not run both connector variants or expect local
   ingress YAML to replace dashboard-managed configuration.
   Route only the API hostname to the tunnel. Keep the API cache bypassed,
   use HTTPS, and avoid browser challenges on API OPTIONS requests. If the
   frontend is proxied through Cloudflare, use Full (strict) TLS to GitHub
   Pages. Do not use Flexible TLS.
8. From the public browser, test session creation, built-in dataset analysis,
   PA → DPD → inference, authenticated downloads, cross-session denial,
   expiry and reset. Keep public routing disabled until these controls and
   the guest reset timer are ready. The local health check is:

   ```bash
   curl -f -H 'Host: 127.0.0.1' http://127.0.0.1:18765/healthz
   ```

The service has no third-party telemetry or mail notifications. Inspect
`systemctl status` and the journal for the VM, connector, guest API and reset
timer. A failed cleanup disables session admission and makes health return
503. Stop the connector to take the public API offline; leave the guest reset
timer running so data is still purged. Source updates require restarting the
guest API, which intentionally invalidates all existing sessions.

### DNS migration and connection failures

A healthy Tunnel does not prove that visitors can resolve the API hostname.
After a nameserver change, some recursive resolvers can still query the old
provider and cache `NXDOMAIN` for an API record created only at Cloudflare.
Compare the client's configured resolver, public resolvers and both providers:

```bash
dig api.opendpd.com A
dig @1.1.1.1 api.opendpd.com A
dig @8.8.8.8 api.opendpd.com A
dig @ali.ns.cloudflare.com api.opendpd.com A
dig @ns01.squarespacedns.com api.opendpd.com A
```

Prepare the required records at both providers before changing nameservers
where possible. If migration has already started, retain the old zone during
the transition and update it where the old provider permits this. Do not
switch nameservers back just to enable editing. Negative answers already
cached by routers or recursive resolvers remain until their TTL expires or
that resolver's administrator clears them; flushing only the browser or OS
cache does not clear an upstream cache.

An `/etc/hosts` override or `curl --resolve` can separate DNS failures from
TLS/API failures, but does not validate public DNS. Repeat the browser session
test with the normal resolver after removing any diagnostic override. A
successful lookup immediately after removal can still be a cached positive
answer; also check the configured upstream resolver directly.

The frontend gives a localized connection error and an explicit Retry action.
It does not automatically retry session creation or other POST requests, or
send a bearer token to an alternate origin. Browser fetch errors cannot by
themselves distinguish DNS, offline, TLS and CORS failures.

## Validation evidence

Local validation on 2026-09-12 passed 279 Python unit tests (1 skipped),
32 public/local Studio integration tests, 178 existing frontend tests and
4 web-client tests. Frontend type checking, lint, production build and npm audit
also passed (zero reported dependency vulnerabilities at that check).

The dedicated offline production VM completed real built-in analysis, PA
training, DPD training and inference. A second session could not read the
first session's dataset, run or events; upload requests were denied. The
daily-reset unit was invoked during an active training run to verify worker
termination, token invalidation and replacement of the tmpfs with an empty
session directory. Source is read-only to the worker, guest Internet egress
is blocked and swap is disabled. Public DNS/Tunnel activation and a browser
test through the real HTTPS hostname are separate release checks.

References: [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[Cloudflare domain onboarding](https://developers.cloudflare.com/fundamentals/manage-domains/add-site/),
[Tunnel routing](https://developers.cloudflare.com/tunnel/concepts/routing/),
[Tunnel origin parameters](https://developers.cloudflare.com/tunnel/reference/origin-parameters/).
