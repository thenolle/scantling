---
title: Configuration and logging
section: Architecture
order: 4
description: How settings are loaded and validated, and how logs are written and redacted.
---

# Configuration and logging

## Configuration

Settings are read in three layers, each overriding the previous one.

1. Built-in defaults.
2. `scantling.config.json`, found in the working directory or given by `SCANTLING_CONFIG`.
3. Environment variables named `SCANTLING_<SECTION>_<SETTING>`, for example `SCANTLING_SERVER_PORT`.

Every value is validated. Unknown keys, wrong types and out-of-range numbers are all reported together with their setting path. Relative paths are resolved against the directory of the configuration file.

```bash
cp scantling.config.example.json scantling.config.json
bun run config
```

| Setting                    | Default     | Notes                                      |
| -------------------------- | ----------- | ------------------------------------------ |
| `server.host`              | `127.0.0.1` | Other hosts need `server.allowRemote`      |
| `server.port`              | `4100`      |                                            |
| `paths.dataDir`            | `data`      | Can point at an external drive             |
| `resources.memoryFraction` | `0.6`       | Share of RAM jobs may use                  |
| `resources.reserveGiB`     | `3`         | Memory kept free for the system            |
| `resources.maxWorkers`     | `8`         | Upper bound for worker threads             |
| `privacy.memoryEnabled`    | `true`      | Conversation memory                        |
| `privacy.trainingAllowed`  | `false`     | Nothing is used for training until enabled |
| `privacy.webSearchEnabled` | `false`     | The assistant stays offline until enabled  |
| `logging.level`            | `info`      | `debug`, `info`, `warn` or `error`         |

> [!NOTE]
> The server listens on the loopback interface only. Exposing it to a network is an explicit opt-in.

## Logging

Logs are structured entries with a time, level, scope, message and fields. They go to the console, to a rotating JSON-lines file under `paths.logDir`, and to an in-memory buffer that the interface reads.

Every entry is sanitised before any sink receives it:

- Fields named like `password`, `token`, `apiKey`, `authorization` or `cookie` are masked.
- Strings that look like GitHub, AWS, Google or Slack credentials, JWTs, bearer tokens, private keys or `password=...` pairs are replaced with `[redacted]`.
- Errors are converted to plain objects, and cycles and very deep structures are cut off.

A sink that fails never interrupts the application. The failure is counted instead.
