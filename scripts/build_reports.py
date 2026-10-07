"""Build the README and compact HTML overview from measured result files."""
from pathlib import Path
import html
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def md_table(frame):
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)

def build():
    result = ROOT / "results"
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(result / "metrics.csv")
    val = metrics[metrics["split"] == "validation"].sort_values("f1_yes", ascending=False)
    columns = ["model", "accuracy", "precision_yes", "recall_yes", "f1_yes", "roc_auc", "average_precision"]
    table = val[columns].copy()
    table.columns = ["Model", "Accuracy", "Precision (yes)", "Recall (yes)", "F1 (yes)", "ROC-AUC", "AP"]
    for c in table.columns[1:]:
        table[c] = table[c].map(lambda v: f"{v:.4f}")
    cv = pd.read_csv(result / "cv_summary.csv")
    cv_table = pd.DataFrame({"Model": cv["model"], "Mean CV F1": cv["mean_cv_f1_yes"].map(lambda v:f"{v:.4f}"),
                             "Fold SD": cv["std_cv_f1_yes"].map(lambda v:f"{v:.4f}"),
                             "Candidates": cv["candidates"], "Folds": cv["folds"]})
    records = summary["split_records"]
    cm = summary["confusion_matrix_validation"]
    jobs = pd.read_csv(result / "eda_job_rates.csv").set_index("job")
    previous = pd.read_csv(result / "eda_previous_campaign_rates.csv").set_index("poutcome")
    campaigns = pd.read_csv(result / "eda_campaign_rates.csv", dtype={"band": str}).set_index("band")
    gaps = pd.read_csv(result / "generalization_gap.csv")
    gap_table = gaps.copy()
    gap_table.columns = ["Model", "Train F1", "Validation F1", "Train minus validation"]
    for column in gap_table.columns[1:]:
        gap_table[column] = gap_table[column].map(lambda value: f"{value:.4f}")
    readme = f"""# Bank Marketing Response

**Machine Learning Algorithms · Midterm Project Report**

**Team:** Nurlan Ramazan · Araizhan Tazhimova

[Notebook](notebooks/bank_marketing.ipynb) · [Python source](src/bank_marketing.py) · [Presentation](presentation/bank_marketing_midterm_redesigned.pptx)

## 1. Introduction

**Project question:** Can client and campaign information available before a
scheduled phone call predict whether a client will subscribe to a term deposit?

This is a binary classification task: `y=yes` is the positive class and `y=no`
is the negative class. The intended application is to help prioritize marketing
contacts. The primary success criterion is an improvement in **positive-class F1**
over a majority-class baseline. Precision, recall, ROC-AUC and average precision
(AP) describe complementary aspects of performance.

## 2. Dataset

The project uses the official [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing),
specifically **bank-additional-full.csv**: **{summary['source_rows']:,} records,
20 input features and one target**. The observations describe telephone campaigns
at a Portuguese bank from May 2008 to November 2010. The dataset is licensed under
**CC BY 4.0**.

The original archive was downloaded from UCI; the full CSV was loaded using a
semicolon separator and retained unchanged. Loading includes a SHA-256 check,
schema inspection and a data-quality audit. No synthetic records are used.
[Source documentation and checksum](data/README.md).

| Feature group | Variables |
|---|---|
| Client | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Current contact | `contact`, `month`, `day_of_week`, `duration`, `campaign` |
| Previous campaign | `pdays`, `previous`, `poutcome` |
| Economic indicators | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |
| Target | `y`: term-deposit subscription (`yes` / `no`) |

`duration` is excluded from modelling because it is known only after the call.
Rows are campaign observations; client IDs and complete timestamps are unavailable.
[Feature definitions](results/data_dictionary.csv).

## 3. Exploratory data analysis

All five EDA plots use the **training partition ({records['train']:,} records)**.
The associations describe this historical sample and do not establish causal effects.

### 3.1 Target distribution

![Training target distribution](results/figures/01_target_distribution.png)

Only **{summary['train_prevalence']:.2%}** of training observations are subscriptions.
Predicting `no` for everyone yields **{1-summary['train_prevalence']:.2%} accuracy**
on train but zero positive-class recall and F1. This imbalance makes accuracy
insufficient for model selection.

### 3.2 Age distribution

![Age distributions within each target class](results/figures/02_age_distribution.png)

The age distributions overlap substantially, so age alone cannot separate the
outcomes. Each class is normalized separately; the vertical axis shows density,
not the probability of subscribing at a given age.

### 3.3 Occupation and response

![Response rates by occupation with group sizes](results/figures/03_job_response.png)

Students have a **{jobs.loc['student', 'response_rate']:.1%}** response rate
(n={int(jobs.loc['student', 'records']):,}), retired clients
**{jobs.loc['retired', 'response_rate']:.1%}** (n={int(jobs.loc['retired', 'records']):,}),
and blue-collar clients **{jobs.loc['blue-collar', 'response_rate']:.1%}**
(n={int(jobs.loc['blue-collar', 'records']):,}). Unequal group sizes and campaign
selection affect the interpretation. [Data](results/eda_job_rates.csv).

### 3.4 Previous campaign outcome

![Response rates by previous campaign outcome](results/figures/04_previous_campaign.png)

Previous campaign success is associated with a **{previous.loc['success', 'response_rate']:.1%}**
response rate (n={int(previous.loc['success', 'records']):,}), compared with
**{previous.loc['nonexistent', 'response_rate']:.1%}** when no previous campaign outcome
exists. This feature describes history preceding the current target.
[Data](results/eda_previous_campaign_rates.csv).

### 3.5 Campaign contact count

![Response rates by recorded campaign contact count](results/figures/05_campaign_contacts.png)

Response is **{campaigns.loc['1', 'response_rate']:.1%}** at the first recorded contact
and **{campaigns.loc['6+', 'response_rate']:.1%}** at six or more contacts. Repeated
attempts may follow non-response; the association does not show that additional
calls reduce interest. The chart includes the current call; preprocessing
converts the counter to earlier contacts. [Data](results/eda_campaign_rates.csv).

## 4. Data cleaning and feature engineering

The training audit found no conventional null values, but six categorical
features contain explicit `unknown` values. Numeric types, plausible ranges,
month codes and weekday codes were checked. Plausible extreme values were retained.
The full source contains **{summary['raw_exact_repeated_rows']} exact repeated rows**;
after removing `duration`, **{summary['repeated_predictor_rows_without_duration']:,}
predictor rows repeat**. These observations were retained and grouped during splitting.
[Quality audit](results/training_data_quality.csv) · [Numeric ranges](results/training_numeric_summary.csv).

| Issue or feature | Treatment | Reason |
|---|---|---|
| `duration` | Exclude | Unavailable at the prediction time |
| Unknown categories | Keep `unknown` as an explicit category | Preserve observations with missing information |
| `pdays=999` | Create `previously_contacted`; replace sentinel with missingness and median-impute elapsed days | Separate no previous contact from a time interval |
| Contact counts | Apply `log1p(campaign - 1)` and `log1p(previous)` | Represent prior contacts and reduce skew |
| Categorical features | One-hot encoding; ignore unseen categories at prediction | Avoid an artificial numeric ordering |
| Numeric features | Standardize for KNN and Linear SVM | Make distances and margins comparable across scales |
| Repeated profiles | Keep identical input profiles in one partition and CV fold | Prevent exact-profile overlap |
| Dates | Retain month and weekday categories | Full dates cannot be reconstructed from the available fields |

Imputation, encoding and scaling are fitted inside each training fold through a
scikit-learn `Pipeline`. The target is never included among the inputs.

## 5. Evaluation design

| Partition | Records | Purpose |
|---|---:|---|
| Train | {records['train']:,} | EDA, preprocessing, cross-validation and tuning |
| Validation | {records['validation']:,} | Model comparison and error analysis |
| Test | {records['test']:,} | Reserved for the final stage; not scored at midterm |

The approximately **60/20/20** split uses `StratifiedGroupKFold` with fixed random
seeds. Groups are identical predictor profiles after excluding `duration`.
The same grouping is enforced in **five-fold cross-validation** on train.
[Partition assignments](results/split_manifest.csv) · [CV folds](results/cv_fold_manifest.csv).

Hyperparameters are selected by mean training-CV F1. Validation then compares the
selected candidates using their default decision rules. ROC-AUC and AP use
continuous scores; AP denotes average precision. No threshold is tuned on test.

## 6. Baseline and models

| Model | Configuration evaluated | Selected configuration |
|---|---|---|
| Majority baseline | Always predict the training majority class | Always `no` |
| Decision Tree | Depth 4/7/10; minimum leaf size 20/80; unweighted/balanced classes | Depth 10, minimum leaf 20, balanced classes |
| KNN | 15/31/61 neighbours; uniform/distance weighting | 15 neighbours, distance weighting |
| Linear SVM | C = 0.01/0.1/1; unweighted/balanced classes | C = 1, balanced classes |

All three lecture algorithms use the same five training folds. The tree requires
no scaling; KNN and Linear SVM use standardized numeric features.
[Selected parameters](results/best_parameters.json).

### 6.1 Cross-validation results

{md_table(cv_table)}

The standard deviation describes variation across folds, not a confidence
interval. These scores are used for tuning and therefore contain selection optimism.
[CV summary](results/cv_summary.csv).

### 6.2 Validation results

{md_table(table)}

**Decision Tree** has the highest validation F1, but its advantage over Linear
SVM is only about **0.002**. This small difference does not establish meaningful
superiority. Linear SVM has higher ROC-AUC and AP, while KNN combines higher
precision with lower recall. [Measured metrics](results/metrics.csv).

![Validation comparison and training cross-validation](results/figures/06_model_comparison.png)

The left panel compares validation precision, recall and F1. The right panel
shows mean training-CV F1 with fold standard deviations.

![Validation ROC and precision-recall curves](results/figures/07_roc_precision_recall.png)

The ranking curves complement the fixed decision-rule metrics. The
precision–recall reference is the positive prevalence in validation.

## 7. Error analysis and model interpretation

![Selected Decision Tree validation confusion matrix](results/figures/08_confusion_matrix.png)

The tree identifies **{cm['tp']:,}** subscriptions and misses **{cm['fn']:,}**,
with **{cm['fp']:,}** false positives and **{cm['tn']:,}** true negatives.
It finds **{summary['validation']['recall_yes']:.1%}** of actual subscribers, while
**{summary['validation']['precision_yes']:.1%}** of its positive predictions are correct.
This illustrates the trade-off between missed subscribers and unproductive contacts.
[Subgroup diagnostics](results/validation_subgroup_errors.csv).

{md_table(gap_table)}

KNN has the largest train/validation F1 gap. Its training queries include their
own neighbours, making the in-sample score particularly optimistic. The tree and
Linear SVM have smaller gaps. [Data](results/generalization_gap.csv).

![Decision Tree feature importance](results/figures/09_tree_importance.png)

`nr.employed` has the highest impurity-based importance in the fitted tree.
Correlated economic variables and split-selection bias limit this interpretation;
importance does not imply causation. [Data](results/tree_feature_importance.csv).

## 8. Conclusion

The three lecture models improve positive-class F1 over the majority baseline.
Decision Tree reaches **{summary['validation']['f1_yes']:.3f} validation F1** and
**{summary['validation']['recall_yes']:.1%} recall**, but its precision shows that
many positive predictions are incorrect. Decision Tree and Linear SVM remain
close candidates for the final stage.

## 9. Limitations and final-stage plan

The data represent one historical banking context. Random partitions mix periods,
so the results do not estimate performance on future campaigns. Grouping identical
profiles does not guarantee client-level separation without IDs. Availability and
publication lags of economic indicators cannot be verified. Validation was used
for model selection; an independent test estimate is still outstanding.

The final stage will:

1. Evaluate temporal stability on development data and audit feature availability.
2. Compare a nonlinear SVM and a wider, bounded hyperparameter search.
3. Select a decision threshold from development predictions using an explicit
   contact budget or error costs.
4. Examine subgroup errors and performance stability.
5. Freeze the pipeline and decision rule, then evaluate the reserved test once.

The current project measures prediction, not the causal effect of calling or
incremental profit. [Detailed final-stage plan](docs/FINAL_STAGE_PLAN.md).

## 10. Team roles

| Member | Proposed technical responsibility |
|---|---|
| Nurlan Ramazan | Data provenance, quality audit, preprocessing, feature engineering and leakage checks |
| Araizhan Tazhimova | EDA, model comparison, cross-validation, metrics and error analysis |
| Both | Interpretation, final-stage planning and presentation |

These are planned roles. Individual technical contributions remain to be
confirmed in the [team contribution log](docs/TEAM.md).

## 11. Reproduction

Tested with **Python {summary['versions']['python']}**. From the project root on Windows:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe scripts/run_notebook.py
.\\.venv\\Scripts\\python.exe scripts/validate_project.py
.\\.venv\\Scripts\\python.exe scripts/build_reports.py
```

On macOS/Linux, use `python3 -m venv .venv` and `.venv/bin/python`.
The raw data are included. Exact dependencies are pinned in `requirements.txt`.
The saved notebook contains executed outputs; running it regenerates the results.
After editing `src/bank_marketing.py`, run `python scripts/sync_notebook.py` before
executing the notebook again.

[Verification record](results/verification.json) · [Full notebook report](results/notebook_report.html) · [Rubric mapping](docs/RUBRIC_CHECKLIST.md)

## References

- Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5K306). CC BY 4.0.
- [Original feature documentation](data/raw/bank-additional-names.txt).
- Scikit-learn documentation: [preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html), [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    caption_map = [
        ("01_target_distribution", "Training class balance", "Only 11.3% of training observations are positive. The majority baseline therefore looks strong on accuracy but has zero positive recall."),
        ("02_age_distribution", "Age distribution", "Class-normalized histograms reveal substantial overlap. Age alone cannot cleanly separate outcomes."),
        ("03_job_response", "Occupation response", "Response differs across occupation groups. Group sizes and campaign-selection effects limit causal interpretation."),
        ("04_previous_campaign", "Previous campaign outcome", "Previous success is associated with a higher response rate. It describes a prior campaign, not the current target."),
        ("05_campaign_contacts", "Contact frequency", "Higher contact counts are associated with lower response here. Repeated attempts may follow non-response; extra calls are not proven harmful."),
        ("06_model_comparison", "Model comparison and CV", "Tree and Linear SVM are close on F1. Error bars show fold SD, not uncertainty intervals for future performance."),
        ("07_roc_precision_recall", "Ranking performance", "ROC and precision–recall curves use continuous scores. AP means average precision, not trapezoidal PR-AUC."),
        ("08_confusion_matrix", "Selected model errors", f"The tree finds {cm['tp']} positive cases, misses {cm['fn']}, and makes {cm['fp']} false-positive predictions."),
        ("09_tree_importance", "Tree explanation", "Impurity importance describes the fitted tree. Correlation and split-selection bias limit interpretation; this is not causal importance."),
    ]
    figure_html = "\n".join(f'<figure><img src="figures/{name}.png" alt="{html.escape(title)}"><figcaption><strong>{html.escape(title)}.</strong> {html.escape(caption)}</figcaption></figure>'
                            for name, title, caption in caption_map)
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bank Marketing Response — Midterm Overview</title>
<style>body{{margin:0;background:#f3f5f8;color:#172738;font:16px/1.6 system-ui,Arial,sans-serif}}main{{max-width:1080px;margin:0 auto;padding:48px 24px}}header{{border-top:7px solid #147d92;padding:24px 0}}h1{{font-size:42px;line-height:1.15;margin:8px 0 16px}}h2{{margin-top:36px;font-size:25px}}.eyebrow{{text-transform:uppercase;letter-spacing:.14em;color:#147d92;font-size:13px;font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}}.card,figure,.panel{{background:white;border:1px solid #dce3e9;border-radius:10px;padding:20px}}.card b{{font-size:30px;display:block;color:#147d92}}.note{{padding:16px 20px;background:#e8f1f3;border-left:4px solid #147d92}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dce3e9}}th{{background:#e5edf2}}figure{{margin:22px 0}}img{{max-width:100%;height:auto;display:block;margin:auto}}figcaption{{margin-top:18px;color:#405367}}a{{color:#086c80}}footer{{font-size:13px;color:#536577;border-top:1px solid #ccd7df;margin-top:36px;padding-top:20px}}@media(max-width:680px){{.cards{{grid-template-columns:1fr}}h1{{font-size:32px}}main{{padding:24px 14px}}}}@media print{{body{{background:white}}main{{padding:0}}figure,.card{{break-inside:avoid}}}}</style></head><body><main>
<header><div class="eyebrow">Machine Learning Algorithms · Midterm</div><h1>Bank Marketing Response</h1><p>Nurlan Ramazan · Araizhan Tazhimova</p><p>Predicting term-deposit subscriptions using attributes available before a scheduled call.</p></header>
<div class="note"><strong>Scope:</strong> historical grouped random validation. The test set remains reserved. The 10-slide presentation is ready. Rehearsal, oral defense and actual student contribution records remain.</div>
<section class="cards"><div class="card"><b>41,188</b>source observations</div><div class="card"><b>{summary['validation']['f1_yes']:.3f}</b>selected validation F1</div><div class="card"><b>{summary['validation']['recall_yes']:.1%}</b>actual positives detected</div></section>
<h2>What was done</h2><p>Five interpreted EDA plots; documented quality checks; duration excluded; domain transformations, one-hot encoding and scaling in fold-local pipelines; majority baseline plus Decision Tree, KNN and Linear SVM; five-fold grouped CV; limited hyperparameter tuning; validation errors and a final-stage plan.</p>
<p>Partition sizes: train {records['train']:,}, validation {records['validation']:,}, reserved test {records['test']:,}. Identical predictor profiles stay together. Client-level independence cannot be verified without client IDs. This random split does not model future campaign performance.</p>
<h2>Measured validation results</h2><div class="table">{table.to_html(index=False,escape=True,border=0)}</div><p>The tree leads by about 0.002 F1 over Linear SVM; no meaningful superiority is established. SVM has higher ROC-AUC and AP. The majority baseline has 88.7% accuracy but finds no positives.</p>
<h2>Training cross-validation</h2><div class="table">{cv_table.to_html(index=False,escape=True,border=0)}</div><p>Selected hyperparameter CV scores have tuning optimism. Fold SD is not a confidence interval. Validation selects the final candidate; no test scoring was performed.</p>
<h2>Evidence and interpretation</h2>{figure_html}
<h2>Reproduce and continue</h2><p>See <a href="../README.md">README</a>, <a href="notebook_report.html">full executed notebook report</a>, <a href="../docs/START_HERE_RU.md">Russian instructions</a> and <a href="../docs/RUBRIC_CHECKLIST.md">rubric mapping</a>. Next: temporal checks, feature-availability audit, nonlinear SVM, cost-aware thresholding and a frozen final-test evaluation.</p>
<footer>Source: <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing">UCI Bank Marketing</a>, Moro, Rita &amp; Cortez (2014), CC BY 4.0. Code and draft explanations prepared with AI assistance; team review and oral defense required. The <a href="../presentation/bank_marketing_midterm_redesigned.pptx">10-slide presentation</a> and <a href="../presentation/SPEAKER_NOTES_EN.md">speaker notes</a> are available separately.</footer>
</main></body></html>'''
    (result / "project_overview.html").write_text(body, encoding="utf-8")
    print("README and compact HTML report generated from saved results.")

if __name__ == "__main__":
    build()
