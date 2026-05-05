# Security Policy

## Reporting A Vulnerability

Please do not open a public issue for vulnerabilities, safety bypasses, leaked credentials, or customer data exposure.

Preferred reporting paths:

- Use GitHub private vulnerability reporting if it is enabled for this repository.
- Otherwise contact the repository owner directly. Add the maintainer security email here before publishing the repository publicly: `[security-contact@example.com]`.

Include:

- A short description of the issue.
- Steps to reproduce, if safe to share.
- Affected files, commands, or configuration.
- Whether the issue could bypass release gates, human approval, or real robot safety checks.

## Safety-Sensitive Behavior

Treat the following as security issues:

- Changes that silently approve hardware-facing actions.
- Changes that bypass `real_robot_gate` or weaken the required human approval.
- Changes that mark missing evidence as a clean pass.
- Commands that could expose customer logs, calibration data, credentials, or private model artifacts.

## Supported Versions

This project has not declared a public release policy yet. Until releases are created, security fixes should target the default branch.
