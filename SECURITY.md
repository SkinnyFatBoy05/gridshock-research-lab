# Security policy

## Supported version

Security fixes are applied to the latest commit on `main`.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability
reporting for this repository and include the affected revision, reproduction steps, impact, and any
suggested mitigation. If private reporting is unavailable, contact the repository owner through the
email address shown on their GitHub profile.

## Security boundaries

- Offline commands read committed, checksum-locked artifacts and require no credentials.
- The live `fetch` path uses public endpoints, bounded date windows, HTTPS timeouts, capped retries,
  and a project-specific user agent.
- Source JSON is validated as data and is never executed.
- No broker integration, order routing, credential store, or live-trading path exists.
- API keys and personal data must not be committed to this repository.

This policy does not make the research proxy suitable for financial decisions or execution.
