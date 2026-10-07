# Course rubric and evidence

This maps the supplied *Machine Learning Algorithms: Midterm Course Project*
brief to the prepared artifacts. Points are the instructor's rubric weights,
**not an awarded or predicted grade**.

| Criterion | Points | Evidence in this project | Remaining human action |
|---|---:|---|---|
| Problem and scope | 10 | Notebook introduction: binary task, `y`, pre-call scenario, F1 criterion, scope and limitations | Explain why the task and metric make sense |
| Dataset understanding | 20 | Notebook §§1, 3–4; official UCI source, license, dictionary, five interpreted EDA plots | Review interpretations and source version |
| Data preparation | 20 | Notebook §§3, 5; quality audit, repeat policy, unknown handling, pdays sentinel, count transformations, one-hot encoding, scaling | Explain each transformation |
| Evaluation design | 15 | Notebook §§2, 6–8; train/validation/test, profile-group separation, fold-local pipeline, five-fold CV, suitable metrics | Explain random-split limitations and untouched test |
| Baseline implementation | 20 | Notebook §§6–9; majority baseline plus Decision Tree, KNN and Linear SVM; actual results and initial error analysis | Rerun and explain errors/trade-offs |
| Presentation and defense | 10 | 10-slide English PPTX, embedded speaker notes, English script and Russian rehearsal guide | Rehearse 10 minutes; both members speak |
| Team process | 5 | Named team, proposed responsibilities, next-stage plan and contribution-log template | Record actual technical contributions; no fabricated claims |

## Required submission artifacts

- [x] Sequentially executable notebook (`.ipynb`) with code, results and prose.
- [x] Equivalent Python analysis script (`.py`).
- [x] README with task, dataset link, setup, current results and next steps.
- [x] At least four meaningful EDA plots, each with an interpretation (five prepared).
- [x] Cleaning documented: types, ranges, repeats, categories, unknowns, dates.
- [x] Multiple justified feature transformations.
- [x] Train/validation/test allocation with leakage controls.
- [x] Baseline and at least two lecture algorithms (three prepared).
- [x] Cross-validation for at least one model (all three lecture models).
- [x] Open problems and final-stage plan.
- [ ] Actual student contribution entries, verified by the students.
- [x] Presentation: [10-slide PPTX](../presentation/bank_marketing_midterm.pptx), with Introduction, Conclusion and a 10-minute speaking plan.
- [ ] Oral defense with both team members.

The brief does not name a submission platform or deadline. GitHub is a project
delivery option, not proof of official submission. Without a defense, the brief
states a 30-point deduction and a maximum midterm score of 70.
