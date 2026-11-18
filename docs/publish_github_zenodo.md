# Publishing an auditable release

These steps should be followed **after** checking upstream data/model/code licences, excluded private files and remaining missing execution assets.

## GitHub (Git command-line method)

1. Create a new **empty** public repository called `ecg-gaze-supervision` at https://github.com/new (do not add a README at creation).
2. From this extracted `ecg-gaze-supervision/` folder, run:

```bash
git init
git add .
git status --short
git commit -m "Initial research code release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ecg-gaze-supervision.git
git push -u origin main
```

3. Confirm `.gitignore` is working: no patient-level clinical CSVs, dataset folders, original images, pretrained model weights, `.h5`/`.npy`/`.npz` files, tokens or `.env` should appear in the staged file list.
4. The executed combined-loss code is now included. Both missing reviewer manifests and the MIT software licence are included. Check any upstream code that was adapted from third parties, and verify the final public tree before describing this as a reproduction release.

## Zenodo DOI

1. Log into https://zenodo.org/ and link GitHub in account settings.
2. In Zenodo's GitHub integration, **enable** the new repository.
3. Once the final code has been committed, publish a GitHub **Release** with a tag such as `v1.0.0`.
4. Zenodo should archive the release and mint its DOI. Confirm the public Zenodo record and insert the exact version DOI in the manuscript Code Availability section and review response. Do not make up a DOI.

Official instructions:
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

You can use GitHub's web upload for these small files, but GitHub's browser has a per-file upload size limit (25 MiB in its published documentation). Do not upload the raw original 48 MB archive, which also includes excluded MIMIC-style CSVs.
