# OpenDPD Studio 2.2.18

This patch strengthens the import, desktop-session, public-service and GPU-worker boundaries reviewed against 2.2.17.

- Dataset specifications describe data only. They cannot replace training arguments, output directories or checkpoint paths. Imported filenames and artifact paths use portable checks for traversal, drive-relative names, UNC paths and symbolic links.
- Experiment imports validate embedded datasets, runs and configurations against the package manifest. They verify raw-data hashes in private staging and roll back newly published directories if the import fails. Archives have a 2 GiB expanded-data limit and retain a disk-space reserve.
- Share packages redact paths inside JSON values, including Windows paths. Leaderboard recomputation checks package hashes, identities and share-package type before importing.
- Desktop bootstrap links expire after 120 seconds and can be used once. Reopening Studio mints a fresh link using its private launcher credential. Browser process arguments contain a private redirect-file URL. New workspaces are private on POSIX; foreign-owned or group/world-writable workspaces are refused.
- Public CSV and metadata exports count toward the heavy-work limit, cache generated CSVs and use a lock per workspace. JSON requests acquire compute/storage reservations after the body arrives. Per-network concurrency limits supplement the global limits. The network-budget table preserves existing networks at capacity; IPv6 budgets use /48 aggregation.
- GPU input archives accept data/artifact layouts only. Container entry points start in safe-path mode outside the writable workspace. Live reads and final output collection use directory descriptors without following links; final collection follows container removal. The agent's service lifecycle is bound to the VM service.
- Installed `opendpd figures replay BUNDLE --out DIRECTORY` and `opendpd figures reproduce BUNDLE --workspace NEW_DIRECTORY` commands read bundles without importing bundled modules. Hashes check integrity against the included manifest; they do not authenticate its author. Direct helper scripts require Python isolated mode (`-I`, including Python 3.10) or safe-path mode (`-P`, Python 3.11+), and an installed OpenDPD.
- Stored plot text and structure are bounded, Plotly labels are escaped, and oversized strings bypass translation-template matching. NumPy inputs are checked before allocation: at most 64 arrays and 256 MiB of expanded data. CSV exports escape formula-like text while keeping numeric values numeric.
- Restricted checkpoint loading now also covers the Volterra benchmark. Deployment lockfiles have a recurring CI audit. Code Owner coverage and the advisory protected-path check cover the additional security boundaries.

## Compatibility and rollout

Existing valid experiments retain their numerical configurations. Files containing unsupported dataset-specification fields, ambiguous paths or inconsistent metadata are rejected; clean data descriptions or re-export the experiment using a matching version. Larger local NumPy captures must be split into bounded captures. Windows/macOS checks are included in the CI definition; local verification for this patch was on Linux.

The source changes and deployment unit templates require an ordinary release deployment to take effect on hosted Studio. The production VM, host services, firewall, GitHub rulesets and Cloudflare settings are not changed by this patch. See the [security review](security-review-2.2.18.md) for the assessment, verification record and remaining boundaries.
