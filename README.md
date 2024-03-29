# **Hi**gh drug **dos**es in patients with **fai**ling **kid**neys (pronounced: high-dose-fake-it)

[![DOI](https://zenodo.org/badge/341922108.svg)](https://zenodo.org/badge/latestdoi/341922108)

**Repo contributors**: [Benjamin Skov Kaas-Hansen](http://github.com/epiben) [code author], [Cristina Rodriguez Leal](http://github.com/crlero) [code review], [Davide Placido](http://github.com/daplaci) [code review], [Hans-Christian Thorsen-Meyer](https://github.com/hcthorsen) [code review]

The repo holds the public version of our full analytic pipeline of the paper. Symlinks and other internal files have been removed as they are non-essential for reading the code in the repo, and because they would not port anyway.

### Publications

- [Kaas-Hansen BS et al. *Using Machine Learning to Identify Patients at High Risk of Inappropriate Drug Dosing in Periods with Renal Dysfunction*. Clinical Epidemiology. 2022;14:213-223. doi:10.2147/CLEP.S344435](https://doi.org/10.2147/CLEP.S344435)
- [Kaas-Hansen et al. *Machine Learning to Identify Patients at Risk of Inappropriate Dosing for Renal Risk Medications: A Critical Comment on Kaas-Hansen et al [Response to Letter]*. Clinical Epidemiology. 2022;14:765-766. doi:10.2147/CLEP.S375668](https://doi.org/10.2147/CLEP.S375668)

### Scope of study
Develop a prediction model for identifying patients at high risk of receiving inappropriate doses of select renal risk drugs.

### Design
Prediction study. Index set at admission time + 25 hours (see paper for justification). Several binary outcomes (levels of number of daily inappropriate doses). 

### Data sources (all Danish)
- National Patient Register
- In-hospital medication data
- In-hospital biochemistry (estimated glomerular filtration rate)

### Model files
The binaries are available in this repository. Feel free to contact the corresponding author for help using them. 

### Software used
The pipeline uses are variety of Python and R libraries. Especially some of the Python modules are a bit dated and so the code might not run with recent versions.

#### R (v3.6.1) packages
- data.table v1.14.2
- lubridate v1.8.0
- patchwork v1.0.0
- plyr v1.8.6
- purrr v0.3.4
- rmda v1.6
- ROCR v1.0-11
- RPostgreSQL v0.6-2
- tableone v0.11.1
- tidyr v1.1.4
- tidyverse v1.3.0

#### Python (v3.8.8) modules
- json v2.0.9
- numpy v1.18.5
- optuna v2.4.0
- pandas v1.3.5
- scipy v1.7.3
- shap 0.40.0
- sklearn v1.0.1
- snakemake v6.0.4
- sqlalchemy v1.4.28
- tensorflow v2.3.1
