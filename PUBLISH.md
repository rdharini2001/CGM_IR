# Publishing the OPTIC paper companion

This project is prepared as the computational companion to the TechRxiv article [doi:10.36227/techrxiv.24465955.v1](https://doi.org/10.36227/techrxiv.24465955.v1).

To publish it in the existing `rdharini2001/CGM_IR` project, review the generated archive and its model figures, then merge its content onto a dedicated Git branch. The `docs` folder is a ready-to-deploy GitHub Pages root and includes the paper, scholarly supplementary methods, static site, and assets. Enable Pages under **Settings > Pages**, source branch `main` with `/docs` after merging. The homepage points to the same repository for code access.

A full run is `bash run_all.sh`. The post-fitting comparison is saved in `results/calibration_coverage.csv` and is never used for optimizing weights. Its 51-entry threshold count consists of 43 directly named metrics plus 8 explicitly labeled accuracy proxies. The website itself presents independent experimental data, calibration, methods, the article, and references without displaying a paper-matching scoreboard.

No remote GitHub branch or live website is changed by generating this archive.
