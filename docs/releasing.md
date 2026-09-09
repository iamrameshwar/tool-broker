# Releasing

Ten packages ship from this repo: `toolbroker` and nine `toolbroker-*` plugins.
They release together, at one version, because a plugin pinned to a core it was
never tested against is the failure mode this monorepo exists to avoid.

## The short version

```bash
# 1. bump every version in lockstep
# 2. commit
git tag v0.1.0 && git push origin v0.1.0
```

The tag runs `.github/workflows/publish.yml`, which refuses to publish anything
until the full suite passes, then uploads the core first and the plugins after.

## PyPI: one-time setup

No API tokens. PyPI's **Trusted Publishing** mints a short-lived credential from
GitHub's OIDC identity, scoped to one workflow file in one repository. There is
no secret in this repo to leak, and nothing to rotate.

For **each** of the ten package names, at
<https://pypi.org/manage/account/publishing/>, add a *pending publisher*:

| Field | Value |
|---|---|
| PyPI project name | `toolbroker`, then each `toolbroker-<plugin>` |
| Owner | `iamrameshwar` |
| Repository name | `tool-broker` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

"Pending" is the important word: it registers the publisher *before* the project
exists, so the very first upload is authenticated the same way as every one
after it. There is no bootstrap step where you paste a token once.

Then create two GitHub environments under **Settings → Environments**, named
`pypi` and `testpypi`. Adding required reviewers to `pypi` is worth it: it makes
publishing a decision somebody makes rather than something a pushed tag does.

Repeat the same registration on <https://test.pypi.org> for the dry-run path.

## Before the first real release

**Drop the `.dev0` suffix.** `0.1.0.dev0` is a development release, and
`pip install toolbroker` will not find it — pre-releases need `--pre`. Anyone
following the README would conclude the package does not exist.

```bash
# in pyproject.toml and all nine packages/*/pyproject.toml
version = "0.1.0"
```

The workflow enforces two things so this cannot go wrong quietly: the git tag
must match the core version, and every plugin must be at that same version.

## Dry run first

Actions → publish → **Run workflow**. That builds everything and uploads to
TestPyPI, where a wrong classifier or a broken README renders somewhere that
does not permanently consume a version number. Then:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ toolbroker
```

The second index matters — TestPyPI does not mirror real dependencies, so
without it `pydantic` cannot resolve.

## Versions are permanent

A version number on PyPI can be deleted but never reused. Yanking hides a
release from resolvers without breaking anyone who already pinned it, and is
the right tool for "that build was broken"; deletion is not.

## conda

You do not host this yourself. Publish to PyPI first, then open a pull request
adding a recipe to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes) —
the recipe points at the PyPI sdist and its checksum. Once it merges,
conda-forge creates a feedstock repository, and a bot opens a version-bump PR
every time you publish a new release to PyPI. The ongoing cost after the first
submission is approving those.

Worth doing when someone asks. A pure-Python package with one dependency
installs fine from pip inside a conda environment, so this is about discovery
and channel policy rather than about it working.

## Everything else

There is no third place. Linux distribution packages (Debian, Fedora, Homebrew)
are maintained by those distributions and are not something a project at v0.1
initiates. Private indexes — Artifactory, CodeArtifact, Azure Artifacts — mirror
PyPI, so an enterprise consuming this through one needs nothing from you.

## After publishing

- Verify the install from a clean environment, not from this repo:
  `uv venv /tmp/check && uv pip install --python /tmp/check/bin/python toolbroker`
- Check <https://pypi.org/project/toolbroker/> renders the README correctly.
- Cut the GitHub release from the tag and paste the CHANGELOG entry.
