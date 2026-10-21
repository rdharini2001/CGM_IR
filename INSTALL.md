# Setup and hosting

Install Python 3.11+ and dependencies with `python -m pip install -r requirements.txt`, then execute `bash run_all.sh`. The first run computes neural models and calibrated predictions and may take a few minutes. The script verifies the full 51-entry **post-training threshold comparison** and runs all tests.

For GitHub Pages, add the archive contents to the existing repository, select **Settings → Pages → Deploy from a branch → main → /docs**. The included `docs/.nojekyll` ensures generated assets are served. The page is offline-readable if `docs/index.html` and its sibling assets are kept together. The paper and supplementary document are linked using relative paths.

The supplementary methods explain the computational extensions beyond the paper [S1-S7], while `references/REFERENCES.md` maps each design choice to external literature. Only `analysis/` contains published model-performance targets; that directory is called *after* fitted scores are generated.

The archive is a prepared research companion. It does not replace or publish to the live GitHub repository automatically.
