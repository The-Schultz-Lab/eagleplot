# EaglePlot desktop app

The Streamlit GUI as a desktop program for Windows, macOS and Linux (roadmap D5). It is
the same app as the [browser version](../web/README.md), packaged with
[`@stlite/desktop`](https://github.com/whitphx/stlite/tree/main/packages/desktop) into
Electron: Python (Pyodide) and every package are bundled, so it needs **no Python install
and no network**.

## Build

Needs Node.js 22 and Python.

```bash
python packaging/desktop/build_desktop.py                # installer for this OS, in packaging/desktop/dist/
python packaging/desktop/build_desktop.py --dir          # unpacked app only (faster, for testing)
python packaging/desktop/build_desktop.py -- --mac dmg --x64 --arm64   # extra electron-builder arguments
```

`build_desktop.py` copies `eagleplot/` and `.streamlit/config.toml` into this folder, runs
`npm ci`, dumps the Pyodide runtime and packages into `build/` (`--pyodide-source DIR`
uses a local Pyodide distribution instead of jsDelivr), lets https links open in the
system browser, and runs electron-builder. The version is EaglePlot's own.

[`.github/workflows/desktop-builds.yml`](../../.github/workflows/desktop-builds.yml) builds
all three on every published release and attaches them to it, and can be started by hand
(Actions → Desktop builds → Run workflow). On Linux it also launches the app and runs
[`scripts/desktop/smoke.js`](../../scripts/desktop/smoke.js).

| OS | File | Admin rights |
| --- | --- | --- |
| Windows | `EaglePlot-<version>-win-x64.exe` (per-user install) | No |
| macOS | `EaglePlot-<version>-mac-arm64.dmg` (Apple Silicon), `-mac-x64.dmg` (Intel) | No |
| Linux | `EaglePlot-<version>-linux-x86_64.AppImage` (make it executable, run it) | No |

## Pins

| Package | Version |
| --- | --- |
| `@stlite/desktop` | 0.105.1 (Streamlit 1.65, Pyodide 314.0.3: the same as the browser build) |
| Electron | 45.0.0 |
| electron-builder | 26.18.0 |

## What the desktop app does differently

The app recognises the desktop build from Electron (`eagleplot.utils.runtime`), because
@stlite/desktop can't pass environment variables to Python.

- **Settings persist:** `config.json` (defaults, presets, workflows) is in the user's app
  data folder (Electron's `userData`, e.g. `%APPDATA%\EaglePlot` on Windows), mounted into
  the app at `/home/pyodide/.eagleplot`.
- **Links** (Help, NIST, GitHub) open in the system browser. @stlite/desktop blocks all
  navigation; `build_desktop.py` patches its Electron main process so that http(s) links
  go to `shell.openExternal` and everything else stays blocked. If a new @stlite/desktop
  changes that code, the build stops with a message saying so.
- **Python runs in a Node.js worker** (`nodeJsWorker`), which is needed to mount host
  folders. It runs only EaglePlot's own code; keep it that way (no pickle loading, no
  user-supplied Python).
- **Not yet:** loading a whole folder (Batch Import, Batch Analysis) needs the paths users
  type (`C:\Users\…`) mapped onto mounted folders: roadmap D5.2. Fetching from NIST is off
  (Pyodide's `requests` support needs the browser's XMLHttpRequest, which Node lacks); the
  page links to NIST instead.

## Unsigned builds: what users see the first time

The installers are not code-signed yet (roadmap D5.4).

- **Windows:** SmartScreen says "Windows protected your PC". Click **More info → Run
  anyway**. Some managed laptops block unsigned programs entirely: use the
  [browser version](https://the-schultz-lab.github.io/eagleplot/app/), which can also be
  installed from Chrome or Edge.
- **macOS:** "EaglePlot can't be opened because Apple cannot check it for malicious
  software". Open **System Settings → Privacy & Security**, scroll down and click **Open
  Anyway** (once). The app is ad-hoc signed, so it runs on Apple Silicon.
- **Linux:** `chmod +x EaglePlot-*.AppImage`, then run it.

## Testing

```bash
python packaging/desktop/build_desktop.py --dir
xvfb-run -a node scripts/desktop/smoke.js packaging/desktop/dist/linux-unpacked/eagleplot
```

The app calls `app.enableSandbox()`, and Chromium refuses to run sandboxed as root: run it
as a normal user. On Ubuntu 24.04 the sandbox also needs
`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`.
