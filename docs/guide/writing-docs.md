---
title: Writing documentation
section: Guide
order: 3
description: How Markdown files become pages and how the site is published.
---

# Writing documentation

Every Markdown file under `docs/` becomes a page. Nothing else needs editing.

## Front matter

```yaml
***
title: Page title
section: Guide
order: 2
description: One-line summary used in search.
***
```

| Key       | Effect                                                     |
| --------- | ---------------------------------------------------------- |
| `title`   | Sidebar and page title. Defaults to the first `#` heading. |
| `section` | Sidebar group. Defaults to the folder name.                |
| `order`   | Position inside the section. Lower comes first.            |
| `draft`   | When `true`, the page is left out.                         |

Files and folders starting with `_` or `.` are skipped. Section order comes from `sectionOrder` in `docs/site.json`.

## Supported Markdown

Headings, emphasis, inline code, links, images, tables, nested and task lists, blockquotes, fenced code with highlighting and `> [!NOTE]` style alerts. Raw HTML is escaped on purpose.

Link to other pages with relative paths such as `[policy](../architecture/dependency-policy.md)`. They become in-site routes.

## Publishing

A push to `master` that changes `docs/` or `website/` runs the Docs workflow. It tests and builds the site, then force-pushes the output to the `website` branch. GitHub Pages serves that branch.

> [!TIP]
> Preview changes with `bun run docs`. Refreshing the browser picks up edits to Markdown files.
