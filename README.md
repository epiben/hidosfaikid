# **Hi**gh drug **dos**es in patients with **fai**ling **kid**neys (pronounced: high-dose-fake-it)

[![DOI](https://zenodo.org/badge/341922108.svg)](https://zenodo.org/badge/latestdoi/341922108)

- **Repo contributors**: Benjamin Skov Kaas-Hansen [code author], [Cristina Rodriguez Leal](http://github.com/crlero) [code review], Davide Placido [code review], Hans-Christian Thorsen-Meyer [code review]

The repo holds the public version of our full analytic pipeline of the paper. Symlinks and other internal files have been removed as they are non-essential for reading the code in the repo, and because they would not port anyway.

### Scope of study
Develop a prediction model for identifying patients at high risk of receiving inappropriate doses of select renal risk drugs.

### Design
Prediction study. Index set at admission time + 25 hours (see paper for justification). Several binary outcomes (levels of number of daily inappropriate doses). 

### Data sources (all Danish)
- National Patient Register
- In-hospital medication data
- In-hospital biochemistry (estimated glomerular filtration rate)
