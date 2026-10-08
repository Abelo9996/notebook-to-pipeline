# Security

notebook-to-pipeline executes code: `capture` runs the notebook and `verify` runs your pipeline,
with the same permissions as your user. Only point it at notebooks and pipelines you would run
yourself.

Reference captures store values as Python pickles (`artifacts/*.pkl`), and `verify` loads them.
Loading a pickle can run code, so only verify against capture directories you created. The JSON
files (`capture.json`, `verify.json`, `report.json`) are plain data and safe to share.

The tool makes no network requests of its own. A notebook or pipeline may.

To report a vulnerability, open a private security advisory on the GitHub repository or email the
maintainer. Please do not open a public issue for it.
