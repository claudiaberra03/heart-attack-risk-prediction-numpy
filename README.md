# Heart Attack Risk Prediction from Survey Data

This project predicts whether a respondent of a large public health survey has had a heart attack or coronary heart disease (the BRFSS variable MICHD), using only their survey answers. The task is a strongly imbalanced binary classification problem evaluated with the F1 score. All models, optimizers and tree ensembles are implemented from scratch in NumPy, as required by the project rules.

The work was carried out as Project 1 of the course *CS-433 Machine Learning* (EPFL, autumn 2026). Predictions were submitted to the course competition on AIcrowd.

**Authors:** Claudia Berra, Paolo Cermelli, Gaia Di Paolo.

## Data

The data come from the 2015 Behavioral Risk Factor Surveillance System (BRFSS) of the US Centers for Disease Control and Prevention, as distributed by the course: 328,135 training rows and 109,379 test rows with 321 columns, of which 8.8% of training rows are positive. The CSV files are not included in the repository and must be downloaded from the course competition page into `dataset/`.

## Method

- **Data preparation.** BRFSS special codes are mapped to missing values or zeros column by column, units are harmonised, skip-logic answers are filled, identifiers and survey-design columns are dropped, and 124 of the 321 columns are kept after an exploratory analysis. Inside each cross-validation fold, missing-value indicators, imputation, one-hot encoding and standardisation are fitted on the training part only.
- **Models.** Thirteen methods are compared on the same folds: the six required least-squares and logistic-regression variants, Newton's method, adaptive gradient methods (including Sophia and UniXGrad), a random forest, gradient boosting, and a hybrid model that fits boosted trees to the residuals of a Newton logistic regression.
- **Selection.** Five stratified folds with a fixed seed. The decision threshold is tuned to maximise F1 on the training part of each fold, because a threshold of 0.5 is far from optimal at 8.8% prevalence.
- **Submitted model.** A weighted average (weights 0.35 and 0.65) of the probabilities of two components: a Newton logistic regression with squared, logarithmic and 20 interaction features, and the hybrid model.
- **Label leakage.** The column `HAREHAB1` asks whether the respondent attended rehabilitation after a heart attack, so it is asked only of positive respondents: all 773 training rows where it was asked are positive. The column reveals the label rather than predicting it, and it is excluded from the submitted model.

## Results

F1 in 5-fold stratified cross-validation (mean ± standard deviation over folds) and on the AIcrowd test set:

| Model | CV F1 | AIcrowd F1 |
| --- | ---: | ---: |
| **Ensemble (submitted), without `HAREHAB1`** | 0.4356 ± 0.0020 | 0.446 |
| Hybrid model alone, without `HAREHAB1` | 0.4338 ± 0.0020 | 0.445 |
| Hybrid model with `HAREHAB1` as a feature (leakage, not submitted) | 0.4413 ± 0.0032 | 0.452 |

- The ensemble improves on the hybrid model by 0.0018 F1 in cross-validation and is better in all five folds.
- Among the single methods, the hybrid model has the highest CV F1 (0.4338), followed by Newton logistic regression (0.4323) and adaptive gradient descent (0.4322). Simple rules based on age group, sex or self-reported general health reach at most 0.309 ([baseline table](results/baselines.md)).
- In the ablation study, removing one-hot encoding costs 0.0068 F1 and removing the boosted trees from the hybrid model costs 0.0015; the column selection has no measurable effect. Using a threshold of 0.5 instead of the tuned one lowers F1 to 0.240 ([ablation tables](results/ablation.md)).

The AIcrowd scores were read from the competition platform and are not stored in the repository. All cross-validation numbers come from the JSON files in `results/`.

## Limitations

- The column lists were chosen by looking at the label on the whole training set and then fixed in every fold, so the cross-validated F1 of the selected pipeline is slightly optimistic. The ablation shows no gain from the selection, but the size of the optimism is not measured.
- The data-preparation ablation uses Newton logistic regression, not the submitted ensemble, because the hybrid model is too slow to retune for every variant.
- The regularisation parameter of the Newton component lies at the edge of the tested grid.
- Cross-validation and AIcrowd scores are different estimates (different samples, and a threshold that transfers differently); they should be compared within each column only.

## Report

The report is [report/report.pdf](report/report.pdf); its LaTeX source and figures are in `report/`.

## Reproduce the submission

The project rules allow only NumPy and the standard library (Matplotlib and seaborn for plots). The pinned versions match the course grading environment (Python 3.9):

```bash
pip install -r requirements.txt
python run.py
```

`run.py` trains both components on the whole training set and writes `submissions/submission.csv`, which should contain the same predictions as `submissions/ensemble_newton_boosting.csv`. A full run took about 11.5 minutes on the machine of one of the authors. The unit tests use small synthetic data and need no dataset:

```bash
pytest tests/
```

Detailed instructions, all hyperparameters, the reference checksum of the submission and the commands for every experiment are in [DETAILS.md](DETAILS.md).

## Repository structure

```text
.
├── implementations.py      # the six required methods, with the signatures fixed by the course
├── helpers.py              # data loading and submission writing provided by the course
├── run.py                  # reproduces the submitted predictions
├── src/                    # preprocessing pipeline, models, cross-validation, metrics
│   └── extra_implementations/   # Newton, adaptive methods, random forest, boosting
├── experiments/            # exploratory analysis and the experiments of every method
├── results/                # JSON results of every experiment, baseline and ablation tables
├── submissions/            # prediction files submitted to AIcrowd
├── report/                 # report (PDF and LaTeX source)
├── tests/                  # unit tests
├── DETAILS.md              # full documentation of the pipeline and the experiments
└── requirements.txt
```

## Original repository

The project was developed in [gaiadp/ML-Project-1](https://github.com/gaiadp/ML-Project-1), which contains the complete commit history of the three authors.
