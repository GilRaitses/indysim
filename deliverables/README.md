This folder holds project deliverables and large artifacts that are not kept in the main repository history.

The GitHub Pages deployment failed because the repository previously contained a broken git submodule entry at:

- deliverables/sensorimotor-habituation-model

That submodule had no URL configured, so automated checkouts that fetch submodules error out.

If you need the sensorimotor habituation model assets, add them here as regular files, or reintroduce a submodule with a valid URL in a tracked `.gitmodules` file.
