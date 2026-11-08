# Vendored skills

Unmodified copies of upstream Claude Code skills, pinned so the guidance matches the library and CLI versions this project uses. Both upstreams are Apache-2.0 with no NOTICE file; each folder carries its upstream license.

| Skill | Upstream path | Pinned commit | Copied |
|---|---|---|---|
| `kaggle-benchmarks` | [Kaggle/kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks) `skills/kaggle-benchmarks/` (branch `ci`) | [`cf26479`](https://github.com/Kaggle/kaggle-benchmarks/tree/cf2647923823f65e440bd11e0789d75cada83729/skills/kaggle-benchmarks) | 2026-11-08 |
| `kaggle-cli` | [Kaggle/kaggle-cli](https://github.com/Kaggle/kaggle-cli) `skills/` (branch `main`) | [`2076e08`](https://github.com/Kaggle/kaggle-cli/tree/2076e087da9fb8186636d2fe4d8c9f8cf6c56ac9/skills) | 2026-11-08 |

`kaggle-benchmarks/LICENSE` and `kaggle-cli/LICENSE.txt` are the upstream repo-root license files at the same commits. Every file was checked against the upstream git blob hashes when copied.

## Package versions

[`requirements.txt`](../../requirements.txt) pins the PyPI releases these commits correspond to:

- `kaggle-benchmarks==0.6.2`: `cf26479` is 5 commits past the `v0.6.2` tag (bug fixes and refreshed model slugs in doc examples), and its `__version__` is still `0.6.2`. The skill's `version: 0.7.0` frontmatter is an upstream label that was already there at `v0.6.2`, so it doesn't mean a 0.7.0 release exists.
- `kaggle==2.2.5`: `2076e08` is 5 commits past the `v2.2.5` tag, and none of them touch `kaggle benchmarks`.

## Rules

- Don't edit the vendored skill folders. Project-specific guidance belongs in the root `CLAUDE.md`.
- `.gitattributes` marks this folder `-text`, so git stores the files byte-for-byte and they stay comparable to upstream.

## Updating

1. Choose the new upstream commit and replace the whole skill folder with the copy at that commit (don't merge by hand). Re-fetch the license from the same commit.
2. Update the commit and date in the table above.
3. Update `requirements.txt` and the list above to match. Find each commit's release from `__version__` at that commit (`src/kaggle_benchmarks/__init__.py`, `src/kaggle/__init__.py`), not from the skill's `version:` label.
