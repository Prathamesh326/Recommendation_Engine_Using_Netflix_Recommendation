%%writefile README.md
# Movie Recommendation System using Surprise

## Introduction

This project focuses on building a movie recommendation system, a type of information filtering system designed to predict the "rating" or "preference" a user would give to an item. Recommendation systems are widely used in various platforms, such as e-commerce, streaming services, and social media, to help users discover items they might be interested in.

In this project, we utilize a dataset that appears to be derived from the original Netflix Prize dataset. This dataset contains user ratings for various movies, which will be used to train and evaluate our recommendation model. The goal is to develop a system that can suggest movies to users based on their past ratings and the ratings of similar users.


## Setup

### Subtask:
Describe how to set up the environment and install the necessary libraries.


**Reasoning**:
Append the setup instructions to the README file.



%%writefile -a README.md

## Setup

To run this project, you need to have Python and pip installed on your system.

You can install the necessary Python libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn
pip install git+https://github.com/NicolasHug/Surprise.git
```

Note that the `surprise` library is installed directly from its GitHub repository to ensure compatibility with newer Python versions.

Depending on your environment and how you access the data, you might need to configure Google Drive access if running in a Colab environment, as shown in the initial notebook code (`/content/drive/MyDrive/Netflix/data.csv`, `/content/drive/MyDrive/Netflix/title.csv`).

## Data loading and exploration

### Subtask:
Explain the data loading process and initial data exploration steps (displaying head, checking data types, shape, and rating distribution).


**Reasoning**:
Append the "Data Loading and Exploration" section to the README file and describe the data loading and initial exploration steps based on the provided code and outputs.



%%writefile -a README.md

## Data Loading and Exploration

The movie ratings data is loaded from the `data.csv` file into a pandas DataFrame named `df`.
python
df = pd.read_csv("/content/drive/MyDrive/Netflix/data.csv")

The first few rows of the DataFrame are displayed to get a glimpse of the data structure and content using the `.head()` method.
python
df.head()

To understand the data types of each column, the `.dtypes` attribute is used. This helps identify if any columns need type conversion for subsequent analysis or modeling.
python
df.dtypes

The shape of the DataFrame (number of rows and columns) is checked using the `.shape` attribute, providing an overview of the dataset's size.
python
df.shape

The distribution of ratings is analyzed by counting the occurrences of each rating value using `.value_counts()` and sorting the results. This gives insight into the popularity of different rating levels. The result is stored in a Series called `ratings`.
python
ratings = netflix_data['rating'].value_counts().sort_index(ascending=True)
ratings

Additionally, the number of unique customers and movies in the dataset are determined to understand the scale of users and items in the recommendation system.
python
netflix_data['customer_id'].nunique()
python
netflix_data['movie_id'].nunique()

## Data preprocessing

### Subtask:
Detail the data preprocessing steps, including dropping the 'date' column, converting the 'rating' column to float, and filtering out movies and customers below a certain benchmark.


**Reasoning**:
Append the data preprocessing section to the README file, explaining each step with code snippets as per the instructions.



%%writefile -a README.md

## Data Preprocessing

Before building the recommendation model, the data undergoes several preprocessing steps to prepare it for the Surprise library and improve model performance.

First, the 'date' column is dropped from the DataFrame as it is not required for the SVD-based recommendation model we are building.python
netflix_data = df.drop('date', axis=1)

Next, the 'rating' column is converted to a float data type. The Surprise library works best with float ratings.python
netflix_data['rating'] = netflix_data['rating'].astype(float)

To focus on more relevant data, we filter out movies and customers with a low number of ratings. This helps in reducing noise and computational cost. We calculate a benchmark based on the 70th percentile of the rating counts for both movies and customers.

The rating counts for each movie are calculated and stored in `movie_data`.python
movie_data = pd.DataFrame(netflix_data['movie_id'].value_counts().sort_index(ascending=True))

The movie benchmark is determined as the rounded 70th percentile of the movie rating counts.python
movie_benchmark = round(movie_data["count"].quantile(0.7))

A list of movies to be dropped (those with rating counts below the benchmark) is created.python
drop_movie_list = movie_data[movie_data['count'] < movie_benchmark].index

Similarly, the rating counts for each customer are calculated and stored in `customer_data`.python
customer_data = pd.DataFrame(netflix_data['customer_id'].value_counts().sort_index(ascending=True))

The customer benchmark is determined as the rounded 70th percentile of the customer rating counts.python
customer_benchmark = round(customer_data['count'].quantile(0.7))

A list of customers to be dropped (those with rating counts below the benchmark) is created.python
drop_customer_list = customer_data[customer_data["count"] < customer_benchmark].index

Finally, the `netflix_data` DataFrame is filtered to exclude the movies and customers identified in the previous steps.python
netflix_data = netflix_data.query(
    "movie_id not in @drop_movie_list and customer_id not in @drop_customer_list"
)

The shape of the DataFrame after trimming is printed to show the reduction in data size.python
print("After trimming the shape is:", netflix_data.shape)

## Model building

### Subtask:
Describe the process of building the recommendation model using the Surprise library, including loading the data into the Surprise format and using the SVD algorithm. Explain the cross-validation results.


**Reasoning**:
Append the "Model Building" section to the README file, explaining the use of the Surprise library, loading data into the Surprise format, using the SVD algorithm, and evaluating with cross-validation.



%%writefile -a README.md

## Model Building

We use the `surprise` library to build the recommendation model. The library provides various algorithms for collaborative filtering.

The preprocessed pandas DataFrame (`netflix_data`) is loaded into the `surprise.Dataset` format, which is required by the library. A `Reader` object is used to parse the DataFrame with the specified rating scale. For demonstration purposes, we use a subset of the data (100,000 rows).
```python
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()
data = Dataset.load_from_df(netflix_data[:100000], reader)
```

The Singular Value Decomposition (SVD) algorithm is chosen for this recommendation model. SVD is a matrix factorization technique commonly used in collaborative filtering. An `SVD` object is instantiated.
```python
model = SVD()
```

To evaluate the model's performance, we use cross-validation with measures like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- **RMSE** measures the square root of the average of the squared differences between predicted and actual ratings. A lower RMSE indicates better prediction accuracy.
- **MAE** measures the average of the absolute differences between predicted and actual ratings. A lower MAE also indicates better prediction accuracy and is less sensitive to outliers than RMSE.

We perform 4-fold cross-validation using the `cross_validate` function.
```python
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=4)
```

The output of the cross-validation shows the RMSE and MAE scores for each fold, as well as the time taken for fitting and testing. The average RMSE and MAE across the folds provide an estimate of the model's performance on unseen data.
```
```


## Making recommendations

### Subtask:
Explain how to use the trained model to make recommendations for a specific user, including filtering the movie title data and applying the prediction function.


**Reasoning**:
Append the "Making Recommendations" section to the README file, explaining how to use the trained model for recommendations, filtering movie titles, predicting ratings, sorting recommendations, and displaying the top results, including relevant code snippets as described in the instructions.



%%writefile -a README.md

## Making Recommendations

After training the SVD model, we can use it to predict ratings for movies a specific user has not yet seen. This allows us to recommend movies that the model estimates the user would rate highly.

To demonstrate, we will generate recommendations for `customer_id` 2632461. First, we need a list of all potential movies to recommend. We use the `df_title` DataFrame, which contains movie metadata. We filter this list to include only the movies that were *not* excluded during the preprocessing step (i.e., those not in `drop_movie_list`), as these are the movies the model was trained on and are considered more relevant for recommendation.
```python
user_2632461=df_title.copy()
user_2632461=user_2632461.reset_index()
user_2632461=user_2632461[~user_2632461['Movie_Id'].isin(drop_movie_list)]
```

For each movie in this filtered list, we use the trained `model.predict()` method to estimate the rating `customer_id` 2632461 would give to that movie. The estimated rating is stored in a new column called `estimation_score`.
```python
user_2632461['estimation_score']=user_2632461['Movie_Id'].apply(lambda x: model.predict(2632461, x).est)
```

We then sort the movies in descending order based on their `estimation_score`. The movies with the highest estimated ratings are considered the top recommendations for the user.
```python
user_2632461=user_2632461.sort_values('estimation_score', ascending=False)
```

The top recommendations are the movies at the head of this sorted DataFrame.
```python
user_2632461.head()
```

This process provides a personalized list of movies that the model predicts the user is most likely to enjoy.
```


## Conclusion

### Subtask:
Summarize the project and potential next steps.


**Reasoning**:
Append the Conclusion section to the README file, summarizing the project and suggesting next steps.



%%writefile -a README.md

## Conclusion

This project successfully demonstrated the process of building a movie recommendation system using the Surprise library. By leveraging the SVD algorithm and preprocessing the dataset to include only movies and customers with a sufficient number of ratings, we were able to train a model capable of predicting user preferences and generating personalized movie recommendations. The cross-validation results provide an initial assessment of the model's performance.

Potential next steps to improve and expand upon this project include exploring other recommendation algorithms available in the Surprise library, such as Non-negative Matrix Factorization (NMF) or various collaborative filtering methods, to see if they yield better performance. Further optimization could involve hyperparameter tuning for the chosen algorithm using techniques like grid search. Utilizing the full dataset rather than a subset would likely lead to a more robust and accurate model. Finally, for a real-world application, the model could be deployed as part of a web service or application to provide recommendations to users in real-time.

## Summary:

### Data Analysis Key Findings

*   The project successfully built a movie recommendation system using the Surprise library.
*   The dataset used was preprocessed by dropping the 'date' column, converting ratings to float, and filtering out movies and customers with low rating counts (below the 70th percentile).
*   The Singular Value Decomposition (SVD) algorithm was used for the recommendation model.
*   The model's performance was evaluated using 4-fold cross-validation, measuring RMSE and MAE.
*   Recommendations for a specific user (customer\_id 2632461) were generated by predicting ratings for movies they haven't seen and sorting by the estimated rating.

### Insights or Next Steps

*   Explore other recommendation algorithms (e.g., NMF) and hyperparameter tuning to potentially improve model performance.
*   Consider using the full dataset for training to build a more robust and accurate recommendation system.
