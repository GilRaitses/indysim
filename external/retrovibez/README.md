This directory is a local-only external dependency.

It used to be tracked as a git submodule, but the repository did not include a tracked `.gitmodules` file with a URL, which breaks automated checkouts when submodules are enabled.

If you need it locally, clone it into this path:

```bash
git clone https://github.com/GilRaitses/retrovibez external/retrovibez
```

