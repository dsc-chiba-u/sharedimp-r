## Submission of sharedimp 0.1.0

This is a new submission.

## R CMD check results

0 errors | 0 warnings | 2 notes

* "New submission": expected for a first release.
* "unable to verify current time": environmental (the check machine cannot
  reach a time server); not related to the package.

Checked with `R CMD check --as-cran` on R 4.4.3 (x86_64, Linux) locally, and
through GitHub Actions on macOS (release), Windows (release) and Ubuntu
(devel, release, oldrel-1).

## Notes for the reviewers

* Examples run in a few seconds in total; the test suite takes about 30 s.
* `ranger` is in Imports because the default learner is a random forest.
  `nnet`, `glmnet`, `ggplot2` and `dplyr` are in Suggests and are guarded with
  `requireNamespace()`.
* The package reimplements a published method (Janssen, Guan and Robeva,
  2023, linked in the Description) and does not claim a new method.
