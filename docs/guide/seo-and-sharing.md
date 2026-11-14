---
title: SEO and sharing
section: Guide
order: 4
description: How each page gets its own URL, metadata, social image and structured data.
---

# SEO and sharing

Every Markdown page is rendered at build time into its own HTML file, for example `guide/getting-started/index.html`. A crawler or link preview therefore sees real content and metadata without running JavaScript. The browser app then takes over for instant navigation.

## What each page gets

- A title of 50 to 60 characters and a description of 120 to 160 characters.
- A canonical URL, robots directives and icons.
- Open Graph and X card tags, including a 1200 by 630 image generated for that page.
- JSON-LD structured data: the site, the article and its breadcrumbs.

The site also gets `sitemap.xml` with last-modified dates taken from git, `robots.txt` and a web manifest.

## Controlling the output

| Front matter  | Effect                                              |
| ------------- | --------------------------------------------------- |
| `title`       | Page heading and sidebar entry                      |
| `seoTitle`    | Replaces the generated search and social title      |
| `description` | Meta description. Falls back to the first paragraph |

The build prints a warning for any page whose title or description falls outside the recommended length.

> [!NOTE]
> GitHub Pages serves a project site below `/scantling/`. Crawlers only read `robots.txt` at the domain root, so submit `sitemap.xml` in the search engine consoles.

## Previewing locally

```bash
DOCS_SITE_URL=http://127.0.0.1:4200/ bun run docs:build
bun run --filter @scantling/website preview
```
