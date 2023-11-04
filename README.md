---

# TripAdvisor Reviews Prediction and Topic Modeling

This project explores TripAdvisor reviews with a focus on predicting ratings from review text utilizing Machine Learning (ML) and Deep Learning (DL) techniques. Beyond rating prediction, the analysis aims to unearth the factors that contribute to a hotel's appeal or lack thereof. By examining prevalent topics in reviews, the intention is to offer actionable insights to hotels and platforms like booking.com on areas for improvement to elevate the customer experience.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cesar1884/NLP_reviews_TripAdvisor.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd NLP_reviews_TripAdvisor
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Setting up the environment**:
Execute the following cells at the beginning of your notebook to ensure the correct directory structure and access to modules:

```python
import os
import sys
os.chdir('..')

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())  # This cell is used to add the parent directory to the path so that the modules can be imported in the notebook. Run this cell only once
```

**Accessing cleaned data**:
The cleaned data is available in the `data/cleaned_tripadvisor_reviews.csv` file. There's no need for additional data cleaning as this file contains pre-processed reviews ready for analysis.


## Usage

### Scripts
1. `scripts/text_classifier.py`:
   - Contains the `TextClassifier` class for easily re-running our model.
2. `scripts/preprocessing.py`:
   - Houses the preprocessing pipeline.

### Notebooks
1. `notebooks/data_exploration.ipynb`:
   - Data exploration.
2. `notebooks/baseline_model.ipynb`:
   - Creation and evaluation of the baseline model.
3. `notebooks/improving_model.ipynb`:
   - Efforts to improve the baseline model.
4. `notebooks/Deep_Learning.ipynb`:
   - Utilizing deep learning for potentially better results.
5. `notebooks/topic_modelling.ipynb`:
   - Topic modelling to discover prevalent themes in reviews.

## Results

![image](https://github.com/cesar1884/NLP_reviews_TripAdvisor/assets/94693373/831ee9f8-61a0-4dd7-bc65-fc2893c6c5cc)

---

