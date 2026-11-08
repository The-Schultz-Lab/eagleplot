# Releasing EaglePlot

A release is one pull request and one button. Everything else runs in GitHub Actions:

| Workflow | On a published release it… |
| --- | --- |
| [`release.yml`](.github/workflows/release.yml) | builds the wheel and sdist from the tag, checks them ([`scripts/release/check_release.py`](scripts/release/check_release.py)), uploads them to PyPI (Trusted Publishing: no token), attaches them to the release, and fills in empty release notes from `CHANGELOG.md` |
| [`desktop-builds.yml`](.github/workflows/desktop-builds.yml) | builds the Windows, macOS and Linux installers and attaches them (about 15 minutes) |
| [`docs.yml`](.github/workflows/docs.yml) | already deployed the website and web app when the release PR was merged |

## 1. The release pull request

On a branch:

1. `eagleplot/__init__.py`: set `__version__` to the new version ([semver](https://semver.org/)).
2. `docs/index.html`: the version badge (`nav__badge`) shows the same version.
3. `CHANGELOG.md`: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (add a short
   summary paragraph), start a new empty `## [Unreleased]` above it, and update the compare
   links at the end (`[Unreleased]` → `vX.Y.Z...HEAD`, new `[X.Y.Z]` → `vPREVIOUS...vX.Y.Z`).
4. Check it:

   ```bash
   python scripts/release/check_release.py --tag vX.Y.Z
   python -m build && python scripts/release/check_release.py --tag vX.Y.Z --dist dist
   ```

   Optionally run **Actions → Release → Run workflow** on the branch: a dry run that builds,
   checks and installs the wheel without publishing anything.
5. Open the PR, wait for CI, merge.

## 2. Publish the GitHub Release

On GitHub: **Releases → Draft a new release**.

- **Choose a tag:** type `vX.Y.Z` and pick *Create new tag: vX.Y.Z on publish*. Target: `main`
  (check that its latest commit is the release PR's merge).
- **Title:** `EaglePlot X.Y.Z`.
- **Notes:** leave empty to have them filled in from `CHANGELOG.md`, or write your own.
- Click **Publish release**. *Save draft* publishes nothing and starts nothing.

Then watch **Actions**: *Release* (about 2 minutes) and *Desktop builds* (about 15 minutes).
When both are green, the release page lists the wheel, the sdist and four installers, and
`pip install eagleplot==X.Y.Z` works.

## If something fails

- **The release check fails** (for example the tag doesn't match `__version__`): nothing has
  been uploaded. Delete the release and its tag on GitHub, fix `main` with a PR, and publish
  again.
- **PyPI upload fails:** re-run the failed job. A version already on PyPI is skipped, not
  overwritten: PyPI never accepts the same version twice, so a broken upload needs a new
  version (`X.Y.Z+1`).
- **A desktop installer fails:** re-run the failed job; it replaces the release's files.

## One-time setup (done once per project)

**PyPI Trusted Publishing.** On [pypi.org](https://pypi.org/manage/project/eagleplot/settings/publishing/),
*Publishing → Add a new publisher → GitHub*:

| Field | Value |
| --- | --- |
| Owner | `The-Schultz-Lab` |
| Repository name | `eagleplot` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

PyPI API tokens are then no longer needed for releases; delete any that were only used for
uploading.

**Optional:** in the repository's *Settings → Environments → pypi*, add yourself as a
required reviewer, so every PyPI upload waits for a click.
