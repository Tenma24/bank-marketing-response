# Speaker notes — Bank Marketing Response

Suggested English script for a 10-minute defense. These notes are also embedded in the PPTX. Rehearse and adapt the wording to your own understanding. Speaking assignments do not document completed technical contributions.

## 1. Introduction

**Nurlan Ramazan · 40 seconds**

Our project studies bank marketing response. The question is whether information available before a scheduled phone call can help predict a term-deposit subscription. We use a binary target: yes means the client subscribed, and no means the client did not. The potential use is to prioritize contacts, but this project evaluates prediction rather than the causal benefit of calling. We compare a simple baseline with three algorithms from the course and select candidates using F1 for the positive class.

**Sources:**

- [bank-marketing-response](https://github.com/Tenma24/bank-marketing-response)
- [bank_marketing.ipynb](https://github.com/Tenma24/bank-marketing-response/blob/main/notebooks/bank_marketing.ipynb)

## 2. Dataset and target

**Nurlan Ramazan · 55 seconds**

The source is the official UCI Bank Marketing dataset. We use the additional-full version, which has 41,188 historical records, 20 original inputs and the response target. We preserve the source file and its license information. The chart uses only the training partition. There are 21,929 negative records and 2,784 positive records, so just 11.27 percent are subscriptions. A model that always answers no looks good on accuracy but identifies no subscribers. That is why positive-class F1, precision and recall matter. Rows are campaign observations, and the lack of client IDs limits our ability to prove independence between clients.

**Sources:**

- [bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [C5K306](https://doi.org/10.24432/C5K306)
- [README.md](https://github.com/Tenma24/bank-marketing-response/blob/main/data/README.md)
- [split_manifest.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/split_manifest.csv)

## 3. Client characteristics

**Nurlan Ramazan · 75 seconds**

These two charts examine client characteristics using training data only. The age chart shows the fraction of each class within ten-year age bands. Normalizing each class separately makes the smaller positive class visible. The distributions overlap, so age alone is insufficient. The occupation chart shows response rates and group sizes in parentheses. Students have an observed response rate of 31.8 percent, while blue-collar clients have 6.9 percent. The student group is much smaller, with 519 observations compared with 5,510. These associations may reflect the bank’s campaign selection and other correlated attributes. We should not interpret them as proof that occupation or age causes a subscription, or that the pattern will hold in future campaigns.

**Sources:**

- [eda_job_rates.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/eda_job_rates.csv)
- [bank-additional-full.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/data/raw/bank-additional-full.csv)
- [split_manifest.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/split_manifest.csv)

## 4. Campaign history and response

**Nurlan Ramazan · 70 seconds**

The fourth and fifth exploratory plots describe campaign history. Clients with a successful previous campaign have a response rate of 65 percent in training, compared with 8.9 percent when there is no previous campaign outcome. Because this refers to a prior campaign, it precedes the current target. The contact-count chart shows a lower observed response rate for records with many attempts: 12.9 percent for the first recorded contact and 5.6 percent for six or more. Repeated attempts may follow previous non-response, so this does not demonstrate that making another call reduces interest. The plotted count includes the current call. In preprocessing, we subtract that call to represent the number of earlier contacts available beforehand.

**Sources:**

- [eda_previous_campaign_rates.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/eda_previous_campaign_rates.csv)
- [eda_campaign_rates.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/eda_campaign_rates.csv)
- [bank-additional-names.txt](https://github.com/Tenma24/bank-marketing-response/blob/main/data/raw/bank-additional-names.txt)

## 5. Data preparation

**Nurlan Ramazan · 60 seconds**

We document data checks and use a reproducible preprocessing pipeline. The most important decision is removing duration because the length of the call is unknown before it occurs. Unknown categories remain explicit rather than causing us to discard records. A pdays value of 999 means no previous contact, so we create an indicator and impute the interval using training data. One-hot encoding handles categories, and we transform contact counts. KNN and SVM also receive scaled numeric variables. The dataset contains 12 exact repeated rows. Without record identifiers we cannot prove they are mistakes, so we retain them but keep identical predictor profiles in one partition. All learned preprocessing is fitted separately inside each CV training fold.

**Sources:**

- [training_data_quality.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/training_data_quality.csv)
- [bank_marketing.py](https://github.com/Tenma24/bank-marketing-response/blob/main/src/bank_marketing.py)
- [common_pitfalls.html](https://scikit-learn.org/stable/common_pitfalls.html)

## 6. Evaluation design

**Araizhan Tazhimova · 60 seconds**

The development process uses about 60 percent training, 20 percent validation and 20 percent reserved test data. We group identical predictor profiles before splitting, using the input features after removing duration. This matters because 2,021 predictor rows repeat after that exclusion. The same grouping also applies inside cross-validation. Hyperparameters are selected using five training folds and positive-class F1, while validation compares the selected candidates. We do not score the test partition at midterm. There are two important limitations: clients with changed attributes cannot be linked without IDs, and random splitting mixes historical periods. Therefore our reported results describe a historical random benchmark rather than a forecast of future campaigns.

**Sources:**

- [split_summary.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/split_summary.csv)
- [cv_fold_manifest.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/cv_fold_manifest.csv)
- [summary.json](https://github.com/Tenma24/bank-marketing-response/blob/main/results/summary.json)

## 7. Baseline and lecture algorithms

**Araizhan Tazhimova · 65 seconds**

Our baseline always predicts the majority class. We then compare three course algorithms: a decision tree, K-nearest neighbours and a linear support vector machine. The tree search tests twelve combinations, and KNN and SVM each test six. Each candidate uses the same five training folds. The selected tree has depth ten, a minimum leaf size of twenty and balanced class weights. KNN uses fifteen neighbours with distance weighting. Linear SVM uses C equal to one and balanced weights. Their selected mean CV F1 scores are approximately 0.446, 0.354 and 0.448. The standard deviations describe variation between folds rather than confidence intervals. Since we select hyperparameters on these scores, they contain some selection optimism.

**Sources:**

- [best_parameters.json](https://github.com/Tenma24/bank-marketing-response/blob/main/results/best_parameters.json)
- [cv_summary.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/cv_summary.csv)

## 8. Validation results

**Araizhan Tazhimova · 70 seconds**

This table compares all candidates on validation data that were not used to fit their parameters. We selected positive-class F1 before comparing the models. Decision Tree has F1 of 0.450, Linear SVM 0.448, KNN 0.373 and the majority baseline zero. The gap between the tree and SVM is only about 0.002, so we cannot claim a meaningful superiority. Linear SVM actually performs better on ROC-AUC and average precision, which evaluate the continuous ranking scores. KNN has higher precision but much lower recall. Average precision, abbreviated AP, is not the trapezoidal area under the precision-recall curve. Finally, we use validation to choose the candidate, so these results are not an unbiased final test estimate.

**Sources:**

- [metrics.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/metrics.csv)
- [validation_predictions.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/validation_predictions.csv)

## 9. Errors in the selected tree

**Araizhan Tazhimova · 65 seconds**

The selected tree identifies 577 actual subscriptions and misses 351. It also predicts a subscription for 1,059 records whose actual outcome is no. This gives recall of 62.2 percent and precision of 35.3 percent. In a targeting scenario, false negatives represent missed positive cases, while false positives can mean unproductive contacts. We have no measured call cost or incremental profit, so we do not claim a financially optimal decision threshold. We also inspect train-versus-validation gaps. KNN has training F1 near 0.972 but validation F1 near 0.373. Its in-sample queries include their own neighbours, so that training score is particularly optimistic. These diagnostics motivate additional validation and threshold analysis at the final stage.

**Sources:**

- [summary.json](https://github.com/Tenma24/bank-marketing-response/blob/main/results/summary.json)
- [generalization_gap.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/generalization_gap.csv)
- [validation_subgroup_errors.csv](https://github.com/Tenma24/bank-marketing-response/blob/main/results/validation_subgroup_errors.csv)

## 10. Conclusion

**Araizhan Tazhimova · 40 seconds**

Our conclusion is that the lecture models improve positive-class F1 over a majority baseline without using call duration. Decision Tree and Linear SVM are very close, so we retain both as plausible candidates. This result remains limited to historical data and a grouped random evaluation. For the final stage, we plan to check temporal stability and feature availability, investigate decision thresholds using development data, and freeze the specification before evaluating the reserved test. The repository contains the full notebook, data documentation and all reported results. Thank you. We are ready for questions.

**Sources:**

- [FINAL_STAGE_PLAN.md](https://github.com/Tenma24/bank-marketing-response/blob/main/docs/FINAL_STAGE_PLAN.md)
- [summary.json](https://github.com/Tenma24/bank-marketing-response/blob/main/results/summary.json)
