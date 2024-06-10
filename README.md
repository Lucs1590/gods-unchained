# Gods Unchained

This project aims to predict the optimal strategy (early or late game) for playing cards in the Gods Unchained trading card game. It uses machine learning to analyze card attributes and classify them into early or late-game strategies.

## Workflow Overview

- Preprocessing (`src.preprocessing.py`):
  - Loads the training data (`data/train.csv`).
  - Converts the `strategy` column to boolean (early/late game).
  - Performs exploratory data analysis (EDA) to understand distributions and correlations.
  - Saves the preprocessed data and artifacts.

- Feature Engineering (`src.feature_engineering.py`):
  - Loads preprocessed data.
  - Engineers new features (log transformations, combined features, comparisons).
  - Encodes categorical features.
  - Performs feature selection using mutual information.
  - Saves the engineered dataset and selected features.

- Model Building (`src.model_building.py`):
  - Loads the engineered dataset and selected features.
  - Trains and evaluates multiple classification models (Logistic Regression, Random Forest, Gradient Boosting, SVC, Linear SVC).
  - Selects the best model based on the KS score (Kolmogorov-Smirnov statistic).
  - Saves the best model and predictions for the test data.

## Dependencies

- Python 3.9
- Libraries listed in requirements.txt (install with `pip install -r requirements.txt`)

## How to Run

### Running the Pipeline Locally

1. Install Python 3.9.
2. Install required dependencies.
3. Set up a virtual environment (optional but recommended).
4. Execute the pipeline step by step using:

```bash
python src/pipeline.py --step preprocessing
python src/pipeline.py --step feature_engineering
python src/pipeline.py --step model_building
```

5. Results will be saved in the `data` and `artifacts` directory.

### Running the API with Docker

1. Build the Docker image:

```bash
docker build -t gods-unchained-api .
```

2. Run the Docker container:

```bash
docker run -p 8000:8000 -t -i gods-unchained-api
```

3. The API will be available at `http://localhost:8000`.

### Running the API with Python

1. Run the following command:

```bash
uvicorn app:app --reload 
```

2. The API will be available at `http://localhost:8000`.

## Running Tests

The tests can be found in the `tests` directory. To run the tests, execute the following command:

```bash
python -m unittest discover tests
```

## Pipeline Deployment

1. Set up the GitHub repository
2. In your GitHub repository, go to the "Actions" tab.
3. Select the "Gods Unchained Deployment Pipeline" workflow.
4. Click the "Run workflow" button and provide a name for the execution.
5. The pipeline will start running, and you can monitor the progress in the Actions tab.
