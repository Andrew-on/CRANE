# CRANE — GitHub repository update

Prepared 29 July 2026. Target repository: <https://github.com/Andrew-on/CRANE>

Everything in this folder is ready to commit as-is. Directory layout mirrors the
repository, so files can be copied straight over the top of a clone.

---

## Why this update is needed

The published repository **cannot run**. Three separate problems:

| file | state in repo | consequence |
|---|---|---|
| `src/M12-94040_2.h5` | **2 bytes** (a bare CRLF) | the trained model is absent — no prediction possible |
| `src/HowellsMain_26Groups_LogTransf_mice.pmm.imp_82FTs.csv` | **2 bytes** | the preprocessed training matrix is absent |
| `src/DIST.org.csv` | **not present** | `CNNpredict.py`, `plotCM.py` and `plotDT.py` all read it; without it the app raises `FileNotFoundError` on the first prediction |

The two stub files were almost certainly truncated by a `.gitattributes` / line-ending
or Git-LFS mishap at some point — both contain exactly `0d0a` and nothing else.

The correct copies in this folder were extracted from the published Docker image
(Zenodo record 16755651), which is the authoritative working version.

Beyond restoring those, this update also applies the manuscript-driven edits and adds
the build files needed to reproduce the container.

---

## Files to upload

### Replace existing files

| file | change |
|---|---|
| `src/CRANE.py` | acronym → *CRaniometric **A**ffi**N**ity Estimator*; forensic-scope statement; terminology; fixes the `sduo docker load` typo |
| `src/impt.py` | **bug fix** — `NA_count` used `len(dd.isna().sum())`, which returns the column count (always 83) rather than the number of missing values, so the imputer ran on every request even for complete specimens. Now `int(dd.iloc[:, 1:].isna().sum().sum())`. Complete specimens go from ~26 s to instant. |
| `README.md` | same acronym and scope changes; DOI links corrected to the **concept DOI** `10.5281/zenodo.15979319` (the badge and citation previously pointed at `…15979320`, the pinned first-version DOI, so they froze on version 202507) |
| `src/M12-94040_2.h5` | restores the 6.0 MB trained model over the 2-byte stub |
| `src/HowellsMain_26Groups_LogTransf_mice.pmm.imp_82FTs.csv` | restores the 3.0 MB training matrix over the 2-byte stub |

### Add new files

| file | purpose |
|---|---|
| `src/DIST.org.csv` | population label ordering — **required at runtime** |
| `src/CRANE_V.py` | variant of the app carried in the image; patched identically for consistency |
| `Dockerfile` | present in the image but never published; enables a reproducible build |
| `environment.yml` | conda specification the Dockerfile consumes |

`CHECKSUMS.txt` lists SHA-256 prefixes and sizes for verification after upload.

---

## Suggested procedure

```bash
git clone https://github.com/Andrew-on/CRANE.git
cd CRANE
git checkout -b restore-runnable-release

# copy this folder's contents over the clone, preserving layout
#   src/…            -> src/
#   README.md, Dockerfile, environment.yml -> repository root

git add -A
git status          # expect 5 modified, 4 new
git commit -m "Restore trained model and reference data; fix imputation bug; publish build files

- Replace 2-byte placeholders for M12-94040_2.h5 and the preprocessed training matrix
- Add DIST.org.csv, required at runtime by CNNpredict.py, plotCM.py and plotDT.py
- Fix NA_count in impt.py: counted columns rather than missing values, so imputation
  ran on every request including complete specimens
- Publish Dockerfile and environment.yml to allow the image to be rebuilt
- Update naming to CRaniometric AffiNity Estimator; state bioarchaeological scope
- Point DOI badge and citation at the concept DOI"

git push -u origin restore-runnable-release
```

Merge to the default branch once checked.

### Before pushing — two things worth confirming

1. **`.gitattributes` / Git LFS.** Something truncated two binary-ish files to a bare
   CRLF. If a `text=auto` rule or a stale LFS configuration is responsible, it will
   truncate them again on the next commit. Check with:

   ```bash
   git check-attr -a src/M12-94040_2.h5
   git lfs ls-files
   ```

   If LFS is in play but objects were never pushed, either install LFS properly or add:

   ```
   *.h5    -text -diff
   *.csv   -text
   ```

2. **Verify after pushing** — re-download and confirm the sizes:

   ```bash
   curl -sL https://api.github.com/repos/Andrew-on/CRANE/git/trees/HEAD?recursive=1 \
     | grep -E 'M12-94040|LogTransf|DIST.org'
   ```

   `M12-94040_2.h5` must read ≈ 6,023,240 bytes and the training CSV ≈ 3,006,242.
   If either is 2 bytes again, the problem is item 1.

---

## Related actions elsewhere

**Zenodo** — upload `D:\CRANE_rebuild\crane-shiny.tar` as a **new version** of the existing
record (not a new record), so the concept DOI `10.5281/zenodo.15979319` continues to
resolve to it and the manuscript citation stays valid. While uploading, edit the metadata:

- Title still reads *"CRANE (CRaniometric ANcestry Estimator) Docker Image"* → **AffiNity**
- Description still says *"Designed as a user-friendly platform for researchers and forensic
  anthropologists"* → the manuscript now states CRANE is **not validated for forensic casework
  involving modern decedents**

**Manuscript** — the Data Availability statement now says the training, cross-validation,
SHAP and missing-data simulation scripts are "available from the corresponding author on
request." Publishing them in this repository would be stronger; if they exist in runnable
form, consider adding a `scripts/` directory in the same commit and amending the statement.

---

## Not included, and why

- `crane-shiny.tar` (13.8 GB) — belongs on Zenodo, not in Git.
- Training / cross-validation / SHAP / simulation code — I have never seen these files;
  they exist neither in the repository nor in the container image.
