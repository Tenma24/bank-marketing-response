# Final-stage plan and open questions

1. **Evaluation target.** The current benchmark measures within-history
   generalization with grouped random partitions. Evaluate temporal stability
   on the development data using ordered blocks. Do not present that experiment
   as a prospective test of the full dataset. A prospective final claim needs a
   separately specified chronological holdout design before model selection.
2. **Feature availability.** Verify campaign-counter interpretation and the
   publication times of economic indicators if original logs/documentation
   become available. Compare a conservative feature set without uncertain
   economic values; exclude call duration in every realistic pre-call model.
3. **Models.** Budget a wider tree/KNN search and compare nonlinear SVM kernels.
   Record time and memory rather than running an unconstrained grid.
4. **Operating threshold.** Obtain a contact budget or relative error costs.
   Select a threshold from training out-of-fold scores; assess on validation
   once. Do not choose thresholds from the reserved test.
5. **Stability and error analysis.** Examine development-period changes and
   confidence/uncertainty, rare groups, unknown categories and correlated
   predictors. Subgroup errors alone do not demonstrate fairness.
6. **Final historical test.** Freeze preprocessing, model, hyperparameters and
   decision rule. Refit on train+validation if planned. Score reserved test once,
   compare to the frozen baseline, and report all results, including failures.
   Further iteration after inspecting it requires acknowledging test reuse.
7. **Team and defense.** Complete actual contribution records and rehearse the
   prepared 10-slide deck for a 10-minute defense with both members speaking.
   Update the slides when new final-stage experiments change the findings.

No synthetic data, deployment, causal profitability claims or perfect-score
guarantees are part of this project.
