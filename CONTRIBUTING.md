# Contributing

Thanks for taking an interest in Scantling.

## Licence and contributions

The project uses the PolyForm Strict License 1.0.0, which does not grant redistribution or derivative-work rights. Open an issue and agree on the approach before sending code. By submitting a pull request you confirm that you wrote the change and allow the maintainer to include it in the project under its licence.

## Ground rules

- No new dependencies. The audit fails the build if one appears.
- No pretrained weights, datasets with private data, secrets or generated checkpoints in the repository.
- Strict TypeScript, ESM, no semicolons, single quotes, tab indentation.
- Keep modules small and cohesive, and document public APIs with JSDoc.

## Workflow

1. Open or claim an issue.
2. Branch from `master`.
3. Run `bun run check` before pushing.
4. Add an entry under Unreleased in `CHANGELOG.md` and update the relevant page in `docs/`.
5. Open a pull request using the template.

## Commit messages

Use the form `type(scope): description` in lower case, without a trailing period, at most 72 characters.

Allowed types: `feat`, `fix`, `docs`, `chore`, `ci`, `build`, `test`, `perf`, `refactor`, `revert`.

Enable the local check with `bun run hooks`. CI applies the same rules to every commit in a pull request.

## Releases

Maintainers run `bun run release <major|minor|patch>`, commit the result as `chore(release): vX.Y.Z` and push to `master`. A workflow then publishes the GitHub release from the matching changelog section.
