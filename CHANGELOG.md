# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Client-rendered documentation site built from the Markdown files in `docs/`.
- Automatic publishing of the site to the `website` branch.

### Fixed

- Dependency audit no longer flags string literals that contain the word import.

## [0.1.0] - 2026-11-14

### Added

- Bun workspace with strict TypeScript configuration.
- Dependency audit covering manifests, imports, lockfile baseline, pretrained-model formats and external AI endpoints.
- Continuous integration on Linux and Windows.
- Release automation driven by the version in `package.json`.
- Commit message linting.
