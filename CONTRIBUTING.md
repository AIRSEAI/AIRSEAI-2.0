<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing

[Back to the main README](README.md)

Every commit must carry a Developer Certificate of Origin (DCO) sign-off, or the pull request's DCO
check fails. The repo ships a hook that adds it for you. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

From then on every `git commit`, from the terminal or an editor, ends with

```
Signed-off-by: Your Name <you@example.com>
```

taken from `user.name` and `user.email`. That email must match the one on your GitHub account.

Check it works:

```bash
git commit --allow-empty -m "test sign-off"
git log -1 --format=%B        # the last line is Signed-off-by
git reset --soft HEAD~1       # drop the test commit, keep your changes
```

The hook does not run on rebase or cherry-pick. To sign commits that are already on your branch,
rewrite them and force-push:

```bash
git rebase --signoff main
git push --force-with-lease
```
