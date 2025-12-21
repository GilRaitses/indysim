This folder is for external dependencies that are kept outside the main repository.

GitHub Pages builds failed because these paths were recorded as git submodules, but the repository does not include a tracked `.gitmodules` file with the required URLs.

Removed submodule entries:
- external/larvatagger.jl
- external/larvaworld
- external/retrovibez

If you want these dependencies locally, clone them into the listed paths. Example URLs that match the repo names:
- larvaworld: https://github.com/nawrotlab/larvaworld
- retrovibez: https://github.com/GilRaitses/retrovibez
