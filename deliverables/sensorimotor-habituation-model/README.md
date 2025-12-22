This folder is intentionally kept small in the main repository.

Historically, `deliverables/sensorimotor-habituation-model` was tracked as a git submodule, but the submodule URL was not recorded in a tracked `.gitmodules` file. That breaks automated checkouts (e.g., CI with submodules enabled).

If you need the sensorimotor habituation model deliverable assets:

- Add them here as regular files (recommended for small, public artifacts), or
- Reintroduce a git submodule by adding a tracked `.gitmodules` entry with a valid `url`.

