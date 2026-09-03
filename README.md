# Longitudinal ADNI, OASIS-3, and AIBL Analysis

This README covers local environment setup, restricted-data placement, and the
notebook execution order.

## Privacy requirement

ADNI, OASIS-3, and AIBL data are restricted. Do not commit or distribute raw
files, processed files, notebooks with stored outputs, reports, results, fitted
models, or archives. The project `.gitignore` excludes these materials, but the
staged-file list must still be reviewed before every push.

## Python 3.10 environment

Create the environment from the project root:

```bash
python3.10 -m venv clinical_env
source clinical_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user \
  --name clinical_env \
  --display-name "Python 3.10 (clinical_env)"
```

Start JupyterLab:

```bash
jupyter lab
```

Select **Python 3.10 (clinical_env)** as the kernel for every notebook.

## Dataset directory rules

Keep original provider downloads under `datasets/` and do not modify them in
place. The scripts search cohort directories recursively, so the original
OASIS-3 and AIBL download subdirectories may be retained.

```text
datasets/
├── adni/
├── oasis3/
└── aibl/
```

Use these locations for generated material:

```text
ProcessedData/   Canonical manifests and analysis-ready data
results/         Generated CSV tables and plots
reports/         Narrative reports and QC handoffs
models/          Local frozen model artifacts
```

Do not place generated files such as participant manifests, cleaned tables,
prediction files, `baseline_subj.csv`, or `conversion_errors.csv` in
`datasets/`.

## Required ADNI files

Place the following files directly under `datasets/adni/`. These exact dated
filenames are used by the current scripts:

```text
All_Subjects_CDR_26Aug2025.csv
All_Subjects_MMSE_26Aug2025.csv
All_Subjects_DXSUM_26Aug2025.csv
All_Subjects_PTDEMOG_26Aug2025.csv
All_Subjects_APOERES_26Aug2025.csv
All_Subjects_GDSCALE_29Sep2025.csv
```

The following ADNI files support optional cognitive, genetic, MRI, white-matter
hyperintensity, and fluid-biomarker analyses and can also remain in
`datasets/adni/`:

```text
All_Subjects_ADAS_26Aug2025.csv
All_Subjects_NPIQ_29Sep2025.csv
All_Subjects_GENETIC_29Sep2025.csv
All_Subjects_UCD_WMH_29Sep2025.csv
All_Subjects_UCSFFSX7_26Aug2025.csv
All_Subjects_UPENNBIOMK_ROCHE_ELECSYS_29Sep2025.csv
```

If newer ADNI exports are substituted, update the explicit filenames in the
scripts before execution and rerun validation from Phase 1.

## Required OASIS-3 files

Keep these files anywhere below `datasets/oasis3/`; recursive lookup is used:

```text
oasis3Subjects.csv
OASIS3_demographics.csv
OASIS3_UDSb4_cdr.csv
OASIS3_UDSd1_diagnoses.csv
OASIS3_UDSb6_gds.csv
OASIS3_UDSc1_cognitive_assessments.csv
OASIS3_Clinical and Psychometrics_dictionary.xlsx
```

Optional imaging and biomarker analyses use these files when available:

```text
OASIS3_Freesurfer_output.csv
OASIS3_amyloid_centiloid.csv
OASIS3_AV1451_braak_tauopathy.csv
OASIS3_AV1451_PUP.csv
OASIS3_AV1451L_PUP.csv
OASIS3_PUP.csv
```

Other provider-supplied UDS, PET, MRI, CT, and metadata CSV files may remain in
their original `datasets/oasis3/scans/.../resources/csv/files/` directories.
They are not a replacement for the required clinical files above.

## Required AIBL files

The original extraction can remain under:

```text
datasets/aibl/aibl_19Sep2019/Data_extract_3.3.0/
```

The current implementation recursively locates these core CSV files:

```text
aibl_cdr_01-Jun-2018.csv
aibl_mmse_01-Jun-2018.csv
aibl_pdxconv_01-Jun-2018.csv
aibl_ptdemog_01-Jun-2018.csv
aibl_apoeres_01-Jun-2018.csv
```

Keep these supporting AIBL files when available:

```text
aibl_adsxlist_01-Jun-2018.csv
aibl_registry_01-Jun-2018.csv
aibl_visits_01-Jun-2018.csv
aibl_neurobat_01-Jun-2018.csv
aibl_labdata_01-Jun-2018.csv
aibl_medhist_01-Jun-2018.csv
aibl_mrimeta_01-Jun-2018.csv
aibl_mri3meta_01-Jun-2018.csv
aibl_av45meta_01-Jun-2018.csv
aibl_flutemeta_01-Jun-2018.csv
aibl_pibmeta_01-Jun-2018.csv
aibl_navmeta_01-Jun-2018.csv
aibl_bslcheck_01-Jun-2018.csv
```

## Notebook running order

Run only after the restricted files have been placed under `datasets/` and the
`clinical_env` kernel is active.

### Data exploration

1. `notebooks/DataExploration/adni/01_adni_demographics.ipynb`
2. `notebooks/DataExploration/oasis3/02_oasis3_demographics.ipynb`
3. `notebooks/DataExploration/aibl/03_aibl_demographics.ipynb`
4. `notebooks/DataExploration/04_synthesize_demographics.ipynb`

### Main analysis

5. `notebooks/Phase1/Phase1_DataFoundationAndEndpointValidation.ipynb`
6. `notebooks/Phase2/Phase2_LandmarksMissingnessAndHarmonization.ipynb`
7. `notebooks/Phase3/Phase3_ADNIModelDevelopmentAndInternalValidation.ipynb`
8. `notebooks/Phase3/Phase3_OutcomeBlindLockRepair.ipynb`
9. `notebooks/Phase4/Phase4_LockedOASIS3ExternalValidation.ipynb`
10. `notebooks/Phase5/Phase5_SecondaryAnalysesAndFinalReporting.ipynb`

Do not start Phase 4 until the Phase 3 lock-repair notebook and its independent
QC have completed successfully. Run notebooks from top to bottom only when a
full rebuild is intended. Existing notebooks may contain saved outputs and are
therefore ignored by Git.

## Before pushing code

Check that only non-sensitive code and setup documentation are staged:

```bash
git status --short --ignored
git diff --cached --name-only
```

Never use `git add -f` to add an ignored dataset, notebook, report, result,
model, archive, spreadsheet, or CSV file.

