# Download EaglePlot

EaglePlot is free and open source ([MIT License](https://github.com/The-Schultz-Lab/eagleplot/blob/main/LICENSE)).
There are three ways to use it. All three keep your data on your own computer.

| | Best for | How |
| --- | --- | --- |
| **Web app** | Trying EaglePlot, or computers where you can't install software | [Open it in your browser](app/). Chrome and Edge can also install it as an app. |
| **Desktop app** | Using EaglePlot offline, without installing Python | Windows, macOS and Linux installers on the [Releases page](https://github.com/The-Schultz-Lab/eagleplot/releases) (from version 3.2.0) |
| **Python package** | Scripts, notebooks, and the full GUI | `pip install eagleplot` ([PyPI](https://pypi.org/project/eagleplot/)), then see [Getting Started](getting-started.html) |

## What each version can do

All three are the same EaglePlot, with the same pages and plots, from the same release.

| | Web app | Desktop app | Python package |
| --- | --- | --- | --- |
| Install | Nothing (or **Install** in Chrome/Edge) | Installer, no admin rights | Python 3.9+, `pip install eagleplot` |
| Download | About 45 MB on the first visit | About 190 MB | A few MB, plus its Python packages |
| Works offline | After one visit, once installed | Yes | Yes |
| Start-up | About 8 s | About 12 s | A few seconds |
| Speed of plots and fits | Slower (runs in WebAssembly) | Slower (runs in WebAssembly) | Full speed |
| Load a whole folder (Batch Import) | No | Yes | Yes |
| Fetch spectra from NIST | Links to NIST | Links to NIST | Yes (`pip install "eagleplot[nist]"`) |
| Change the app's theme | No | No | Yes |
| Python API, notebooks, `eagleplot` command | No | No | Yes |
| Settings and presets kept | In the browser | Yes | Yes |
| Data stays on your computer | Yes | Yes | Yes |

The version number is shown on the Home page of all three.

## Desktop app

Download the file for your system from the [latest release](https://github.com/The-Schultz-Lab/eagleplot/releases/latest):

| System | File | Admin rights |
| --- | --- | --- |
| Windows | `EaglePlot-<version>-win-x64.exe` | Not needed |
| macOS (Apple Silicon) | `EaglePlot-<version>-mac-arm64.dmg` | Not needed |
| macOS (Intel) | `EaglePlot-<version>-mac-x64.dmg` | Not needed |
| Linux | `EaglePlot-<version>-linux-x86_64.AppImage` | Not needed |

Python and every package EaglePlot needs are included, so the app works without a network
connection. Settings are kept between sessions.

**The first time you open it:**

- **Windows:** if Windows shows "Windows protected your PC", click **More info → Run anyway**.
  Some managed (university or company) laptops block unsigned programs entirely: use the
  [web app](app/) instead, which Chrome and Edge can also install.
- **macOS:** if macOS says it "cannot check EaglePlot for malicious software", open
  **System Settings → Privacy & Security**, scroll down and click **Open Anyway** (once).
- **Linux:** make the file executable (`chmod +x EaglePlot-*.AppImage`), then run it.

## Code signing

The desktop installers are **not code-signed yet**, which is why Windows and macOS warn
the first time you open them (see above). Every installer is built from this
repository's source by its GitHub Actions workflows and attached to the
[release](https://github.com/The-Schultz-Lab/eagleplot/releases) by them, with no manual
step in between. Download EaglePlot only from the Releases page or this website.

## Privacy policy

EaglePlot does not collect, store or send any information about you or your data.
It will not transfer any information to other networked systems unless you specifically
ask it to.

- **Desktop app:** makes no network requests. Links you click open in your web browser.
- **Web app:** a static site hosted on [GitHub Pages](https://docs.github.com/en/pages).
  Your browser downloads the app, and the Python runtime from the
  [jsDelivr](https://www.jsdelivr.com/) CDN, and runs it locally. Files you open are read in
  your browser and never uploaded. GitHub and jsDelivr may log requests (such as IP
  addresses) as described in their own privacy policies.
- **Python package:** contacts other servers only when you ask it to, for example to look
  up a spectrum in the [NIST Chemistry WebBook](https://webbook.nist.gov/). The GUI turns
  off Streamlit's usage statistics in its settings file (`.streamlit/config.toml`), which
  applies when you start it as described in [Getting Started](getting-started.html) (with
  `launch.bat`, `launch.command` or `streamlit run eagleplot/Home.py` in the EaglePlot
  folder).

Questions: open an [issue](https://github.com/The-Schultz-Lab/eagleplot/issues).
