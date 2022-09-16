# Twitter Sentiment Project
<p align="center"><img src="https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/Tweet%20Emotions.jpeg" width="500" height="300">

</p>This project is an attempt to build a sentiment classifier for Tweets. It is much a preprocessing project as anything else. The project centered around creating a model that could detect the sentiment of Tweets and predict on a test set. 

# Business Problem

The goal of this project was to create a classifier that could attempt to identify negative tweets at a live event. The hope is that the classifier could aid a social media manager in responding to a criticism or a crisis in real time. A nimble and quick classifier would be able to take in a bulk amount of tweets, process them, and then identify the offending tweets. Social media managers could then assess the situation and respond to users in a timely fashion.

# The Dataset
![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/TargetBreakdownRaw.png)
![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/SentimentAfterProcessing.png)

The dataset for this project was tweets from South by Southwest in 2011. Due to initally low scores for the model I augemented this dataset with data from Crowd Flower, another dataset. This was neccesary as the initial dataset was highly imbalanced and needed more negative examples to butress the models training data.

# Processing

The tweets had to go through thoughrough preprocessing for the modeling. The pre-processing process is in itself a place of measurement and decision points. The tweets had to have odd characters and emojis removed as well as HTML characters. Various punctuation was stripped, whitespaces removed, text had to be lower-cased. For the best performances I relied on the sklearn TFIDF vectorizer's processing after I had done intitial preprocessing.

## Brand Processing
![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/SentimentbyProductPreProcessed.png)
As mentioned above the dataset is divided into Sentiment, Object of Sentiment, and Tweet Text. The Object of Sentiment or Product was not used in the modeling, but I thought it would be helpful to identitfy relevent tweets. I was able to expand the Product column greatly, eliminaiting irrelevent tweets from the model. 

## Model Target

I created a binary classifier for this project, for future work it would be interesting to expand -- but aiming to identify negative tweets, I combined the positive and neutral labeled tweets into one target. This essentially divides the model target into Negative and Positive/Neutral. Better scores could certainly been achieved by dropping a Neutral targets, but in a real world situation we would not have the luxery ahead of time of knowing which tweets had which sentiment.


![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/BinaryBeforeNegative.png)
![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/BinaryAfterNegative.png)

# Scoring

I looked to recall score as the guiding metric for this project with an eye to precision. As in previous projects I employed a custom Beta-2 scorer to achieve optimal recall without sacrificing precision. I used stratified 5-fold cross validation on the training data to understand overfitting. I also split the data into training/validation/testing. I scored each model on the validation set if it seemed promising on the cross-validation. After making my final model selection I tested it on the testing data. 

# Repository 

## Notebooks 

* [EDA Notebook](Project_Notebooks/EDA_and_PreProcessing.ipynb)
* [Conventional Models](Project_Notebooks/Modeling_Notebook.ipynb)
* [Neural Network Models](Project_Notebooks/Neural_Network.ipynb)
* [Final Model Selection and Interpretation](Project_Notebooks/FinalNotebook.ipynb)

## Data

The data folder contains the inital Tweets for the project as well as the supplemental Tweets. There is a GitIgnore for the Glove Vectors that are too large for Github and are readily available elsewhere. 

## PDFs

The PDFs folder contains copies of my work as well as my presentation slides.

## Images

The image folder contains charts and images used for the project and the presentation.

## Saved Models

I saved some particularly long grid search models and saved my final model as well as my final LSTM model.

### EDA Notebook

The EDA/Preprocessing notebook contains an exploration of the dataset as well as work to eliminate nulls, populate the product column and ready the text for modeling. The Modeling notebook contains conventional models where I experimented with Bag-of-Words vs TFIDF. I attempted several different models including Logistic Regression, Naive Bayes, Support Vector Machines, and Random Forrest. I tracked the model scores and then added the additional negative tweets when I encountered low scores on cross-validation and the validation set. 

### LSTM Neural Network Model

I created another notebook to work with LSTM neural networks for the first time. These models almost acheived as good scores as my best conventional model and I will look to working with them in the future, however, I was unable to perform cross-validation on these models, and the scores of the best Logistic Regression compared with my best LSTM model favorably.

### Final Model Selection and Interpretation 

I summarized my work in the final notebook and it has links to the other notebooks along with exploration of the final model. I epxlored the top features and produced wordclouds for each of the important brands.

![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/FinalConfusionMatrix.png)

My final model ended up being arguably the simplest model - a Logistic Regression using Bag-of-Words using unigrams achieved the best scores. Perhaps its simplicity was its strength. It scored over 80% on recall during cross-validation and over 83% on the final test set. From this model I was able analyze the frequencies for relevent products. 

![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/PositiveCoefficients.png)
![alt text](https://github.com/seanisthegood/Twitter_Sentiment_Project/blob/main/Images/NegativeCoefficients.png)
# Further Work

* Increase Dimensions with Glove Model

I only used the pretrained glove vectors with 50 dimensions, I may have achieved better scores with increased dimensions. Training the neural networks is a delicate process and interpretability requires more work. I opted for the simplier model at this time as the LSTM model did not achieve dramatically better scores.

* Other Vectorization Choices

Vader vectorization may be a worth exploring as it a sentiment scoring that is aimed at assessing sentiment in a social media setting.

* Experiment with Different Pre-processing

As I continue NLP work the challenges I forsee are misspellings, saracasm, jargon, and image usage. This is an older dataset where GIF usage had not yet become as widespread. Expression is constantly evolving online. 

* Halving Grid Search Useage

I employed Grid Search -  it is slow and blunt. Sklearn's Halving Grid Search was employed as my impatience grew, but I encoutered some errors. It saves time and my all indication scores comperably. 
