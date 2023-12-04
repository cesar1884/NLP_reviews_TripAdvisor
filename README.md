# TripAdvisor Reviews Prediction and Topic Modeling

![image](https://github.com/cesar1884/NLP_reviews_TripAdvisor/assets/94693373/a0f0fda3-a353-48f1-80e6-81d5afdee8cb)


This project explores TripAdvisor reviews with a focus on predicting ratings from review text utilizing Machine Learning (ML) and Deep Learning (DL) techniques. Beyond rating prediction, the analysis aims to reveal the factors that contribute to a hotel's appeal or lack thereof. By examining prevalent topics in reviews, the intention is to offer actionable insights to hotels and platforms like booking.com on areas for improvement to elevate the customer experience.

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

The primary metric of interest was not accuracy, but the recall for classes 2 and 3, and the overall F1 score, due to the unequal data distribution among classes.

## Problem Statement

The main challenge encountered with this dataset is achieving a sufficient recall for classes 2 and 3, which contain relatively fewer data compared classes 3 and 4 (class 1 contains few datas but the model is generally able to classify them).

## Initial Approach with Classical Machine Learning

# ***just change result from there***


### Experimentation and Results 
- **Techniques Employed**: To counter the class imbalance, `class_weight` was tried. Tried some processing techniques as bi grams, remove rare and common words. hyperparameter tuning
- **Model Employed**: The model that yielded the best results is a text classifier based on Logistic Regression with a `TfidfVectorizer` as a vectorizer.
    ```python
    classifier = TextClassifier(
        model=LogisticRegression(
            max_iter=100, 
            class_weight='balanced', 
            penalty='l2', 
            C=0.1, 
            multi_class='ovr',
            fit_intercept=False, 
            solver='newton-cg'
        ), 
        vectorizer=TfidfVectorizer(
            max_features=20000, 
            stop_words='english'
        )
    )
    ```
- **Results**: 
    - Accuracy: 61%
    - Recall for Class 2: 48%
    - Recall for Class 3: 44%
    - Global F1 Score: 55% (macro avg)

## Exploration with Deep Learning

### Challenges Encountered
- Moving to deep learning revealed an overfitting problem, likely exacerbated by the lack of data.

### Model Employed and Results
- A sequential model with an embedding layer, Bi-LSTM, dropout, and a dense layer had the best results.
    ```python
    model = Sequential([
      Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
      Bidirectional(LSTM(64)),  # Bi-LSTM
      Dropout(0.2),
      Dense(5, activation='softmax')
   ])
    ```

## Class Remapping and Final Results

To simplify the problem and achieve better results, class remapping was performed as follows:
```python
mapping = {1: 'negative', 2: 'negative', 3: 'negative', 4: 'good', 5: 'excellent'}

```
- It makes senses as from a Rate of 3 people will start to be mefiants while looking for an hotel on their phone. 
- This remapping allows to approximately equalize the number of instances in each category, which will facilitate the interpretation of the model's outputs. However as we seen in EDA it will probably lead to a lot of misclassification between Good and Negative as 4 was often miscclasified with 3


- **Results with Remapping**:
    - Accuracy: 72%
    - Recall for 'good' category: 61%
    - Global F1 Score: 72% (macro avg)
 
  These results are achieved with the DL model presented just before and after creating a set of classification rules based on probabilities given by softmax ( see more at the end of the notebook "Deep Learning"

### Conclusion

Class remapping led to better results in terms of Accuracy and macro F1 score. Deep learning learning techniques ultimately got result best result in this scenario, but is not really outperforming the ML model. By working more on the ML model we may have as good result.


# Topic modelling

![image](https://github.com/cesar1884/NLP_reviews_TripAdvisor/assets/94693373/831ee9f8-61a0-4dd7-bc65-fc2893c6c5cc)

While looking at these 4 topics we can try to understand how they were created:
- topic 1 is related to resort, "dream holidays hotel", all inclusive, ... this type of hotels (interesting to see the importance of bigram there with punta_cana) there the thingss that interest people is the beach, bar, food and the pool. 
- topic 2 is related to city hotel. there people are more focused about the location, the staff and the confort
- topic 3 is not really clear as it contains a lot of verbs that does not give insight ( we could have removed them ?) but an interesting thing is the fact that it is the only topic that cntains a negative word "bad"
- topic 4 really focus on the room. on this type of hotel we can clearly say that people are mainly interested about the quality of the room


# LDAvis (click on the link below)

[Visualisation LDAvis](https://cesar1884.github.io/NLP_reviews_TripAdvisor/LDAvis.html)

The left part shows the distance between different topics. the size of the circle indicate if lot of review are related to this topic. topic 1 and 3 that correspond to topic 2 and 4 of our precedent plot are really close to each others.

the right part show the top 30 terms by saliency. saliency measure the number of apparition of a words in the review bt also measure the apparation in this topic comparated  to the others topic: a words that appears a lot in this topic but not much in the others will have a high saliency. And it is an incredible insight for us to us as it allows us to measure the importance of a word for a category of hotel.

This way we can say that for each type of hotel that correspond to a particular topic. thy should focus on the words with high saliency. The more they considered them as part of a "topic" the more they have to focus on these subject. 

Adjust the parameter λ modify the weight of the distictiveness, and this is so usefull to establish our hotel type classes. with a low  λ  distinctuveness is high and then we clearly noitce that 1 and 3 are city hotel. 1 contain a lot of cities with a lot of offices and focus on the location comparated to the station and things like that so we can consider them as business hotel while 3 focus on touristic cities and touristic place so we can considere them as city trip hotel.

With this λ topic 4 confirm our observation that it was the only one that contained a negative words as it seem now to contains a lot of negative words. we can consider them as bad hotel even if it is a really different topic than the others but it is an helpful topic for our study. 
An hotel has to do all possible effort to be out of this topic. We have there a list of the things that customers hate the most in what they consider as a bad hotel. this way you know that if you do not focus on these subject you will be classified as a bad hotel


