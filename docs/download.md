# Download EaglePlot

EaglePlot is free and open source ([MIT License](https://github.com/The-Schultz-Lab/eagleplot/blob/main/LICENSE)).
There are three ways to use it. All three keep your data on your own computer.

| | Best for | How |
| --- | --- | --- |
| **Web app** | Trying EaglePlot, or computers where you can't install software | [Open it in your browser](app/). Chrome and Edge can also install it as an app. |
| **Desktop app** | Using EaglePlot offline, without installing Python | Windows, macOS and Linux installers on the [Releases page](https://github.com/The-Schultz-Lab/eagleplot/releases) (from version 3.2.0) |
| **Python package** | Scripts, notebooks, and the full GUI | `pip install eagleplot` ([PyPI](https://pypi.org/project/eagleplot/)), then see [Getting Started](getting-started.html) |

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
  Signed installers (see below) should not show this warning once they have built up a reputation with Microsoft.
- **macOS:** if macOS says it "cannot check EaglePlot for malicious software", open
  **System Settings → Privacy & Security**, scroll down and click **Open Anyway** (once).
- **Linux:** make the file executable (`chmod +x EaglePlot-*.AppImage`), then run it.

## Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io/), certificate by
[SignPath Foundation](https://signpath.org/).

The Windows installers are signed with this certificate. Signing starts with the first
release after the project's approval. Earlier installers are unsigned.

Only files built from this repository's source by its GitHub Actions workflows are
signed. Every change reaches the `main` branch through a reviewed pull request, and
each signing request is approved by hand.

**Team roles**

- **Committers and reviewers:** members of the
  [Schultz Lab](https://github.com/The-Schultz-Lab) at North Carolina Central University
  and other [contributors](https://github.com/The-Schultz-Lab/eagleplot/graphs/contributors).
  Changes are reviewed in pull requests before they are merged.
- **Approvers:** Jonathan D. Schultz ([@NCCU-Schultz-Lab](https://github.com/NCCU-Schultz-Lab)).

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
  up a spectrum in the [NIST Chemistry WebBook](https://webbook.nist.gov/). Usage
  statistics in the Streamlit interface are turned off.

Questions: open an [issue](https://github.com/The-Schultz-Lab/eagleplot/issues).
