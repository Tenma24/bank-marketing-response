# Data source and acquisition

- Dataset: **Bank Marketing**, UCI Machine Learning Repository.
- Exact version used: `bank-additional-full.csv` (41,188 rows, 20 inputs + `y`).
- Landing page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Official download: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- Within that archive: `bank-additional.zip` → `bank-additional/bank-additional-full.csv`.
- Original documentation: [bank-additional-names.txt](raw/bank-additional-names.txt).
- License: **Creative Commons Attribution 4.0 International**, https://creativecommons.org/licenses/by/4.0/.
- Source metadata and checksums: [source.json](source.json).
- Acquisition for this project: 7 October 2026, from the official UCI archive.

Attribution: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

Related study: S. Moro, P. Cortez and P. Rita (2014). *A Data-Driven Approach to
Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22–31.
https://doi.org/10.1016/j.dss.2014.03.001.

The raw CSV is included unchanged for offline reproducibility. It is read with
`sep=";"`. If missing, the notebook downloads exactly the two specified source
files, without extracting arbitrary archive paths. A SHA-256 check detects a
different or modified CSV. No synthetic records are added.

CSV SHA-256:
`74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.

## Interpretation and limitations

The source concerns historical telephone campaigns at a Portuguese bank.
Rows are anonymized campaign/contact observations, not guaranteed unique clients.
The full file is ordered by date (May 2008–November 2010), but full timestamps
and client IDs are absent. The labels describe deposit subscriptions, not the
causal effect of a phone call. There are explicit `unknown` categories even
though conventional empty cells are absent in this version.

`duration` is unavailable before a call and is excluded. `pdays=999` means no
previous contact, not an elapsed interval of 999 days. `campaign` includes the
recorded call and is transformed to the number of earlier campaign contacts.
Economic features are treated as available at the recorded call; publication
lags/revisions cannot be verified from the file and limit operational claims.

The derived data dictionary, training quality audit and exact split manifests
are in `results/`. The random benchmark mixes historical periods; it is not a
future-campaign validation. Repeated feature profiles are grouped to prevent
their crossing partitions, but repeated-client leakage cannot be ruled out.
