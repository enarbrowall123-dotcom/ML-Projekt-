# Student Grade Prediction

Midterm for the ML Zoomcamp 2026.

The idea is simple: can we guess a student's final grade at the start of the school year, before any tests have been written? If so, teachers could spot students who might need extra help early, instead of waiting for the first bad grades.

## Data

I used the [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) dataset from UCI (CC BY 4.0). It has 649 students from two Portuguese schools taking a Portuguese language course. For each student there is information about family, school, study habits, free time, alcohol use and so on, plus three grades: G1, G2 and G3 (final grade, 0 to 20).

The file is included in the repo as `data/student-por.csv`.

## What I found

All the details are in `notebook.ipynb`, but two decisions shaped the project:

1. I dropped G1 and G2. They are the grades from the first two periods and correlate very strongly with the final grade (0.87 and 0.93). With them the model would basically just say "you'll get what you got last time", and they aren't known at the start of the year anyway.

2. I removed the 15 students with a final grade of 0. They all had normal grades earlier in the year (G1 between 4 and 11), so a 0 at the end most likely means they dropped out or skipped the final exam. That's a different problem from predicting a grade.

Without the period grades, no single feature is a strong predictor. The most useful ones were past failures, whether the student wants to go on to higher education, the parents' education, study time and alcohol use.

## Model

I tried linear regression (Ridge), a decision tree, a random forest and XGBoost, tuned on a validation set. Ridge and XGBoost came out on top, so I compared them with 5-fold cross-validation:

| Model | RMSE |
|---|---|
| Predict the mean (baseline) | 2.72 ¬± 0.18 |
| Ridge, alpha = 10 | 2.25 ¬± 0.19 |
| XGBoost | 2.27 ¬± 0.16 |

They're basically tied, so I went with Ridge since it's simpler and easier to explain. On the test set it got an RMSE of 2.14, so predictions are about 2 grade points off on average. Not perfect, but good enough to flag students who are likely to end up below the passing grade of 10.

## Running it

You need Python 3.12 and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```bash
uv sync
uv run python train.py      # trains the model and saves model.bin
uv run python predict.py    # starts the API on port 9696
```

Then in another terminal:

```bash
uv run python test.py
```

`model.bin` is already in the repo, so you can skip `train.py` if you just want to try the API.

### With Docker

```bash
docker build -t student-grade .
docker run -it --rm -p 9696:9696 student-grade
```

## API

Send a POST to `/predict` with the student's data:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "school": "gp", "sex": "f", "age": 16, "address": "u",
    "famsize": "gt3", "pstatus": "t", "medu": 4, "fedu": 3,
    "mjob": "teacher", "fjob": "other", "reason": "reputation",
    "guardian": "mother", "traveltime": 1, "studytime": 3,
    "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes",
    "internet": "yes", "romantic": "no", "famrel": 4, "freetime": 3,
    "goout": 2, "dalc": 1, "walc": 1, "health": 5, "absences": 2
  }'
```

Response:

```json
{"predicted_grade": 14.77, "at_risk": false}
```

`at_risk` is true when the predicted grade is below 10. There's also `GET /health`, and FastAPI's interactive docs at `/docs`. Invalid input (like `"age": 50`) gets a 422 error.

## Files

- `notebook.ipynb`: EDA, model comparison and tuning
- `train.py`: trains the final model
- `predict.py`: the FastAPI service
- `test.py`: sends an example request
- `model.bin`: the trained model
- `Dockerfile`, `pyproject.toml`, `uv.lock`

## Limitations

The dataset is small and comes from two schools in Portugal back in 2005 to 2006, so I wouldn't expect the model to work well elsewhere without retraining. It also says nothing about who is likely to drop out, since those students were removed.
