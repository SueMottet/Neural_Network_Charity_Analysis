# Neural Network Charity Analysis
This data analysis applies machine learning and a neural network also known as a deep learning model to create a binary classifier that is capable of predicting whether applicants will be successful if funded by a charitable organization.
## Overview

A CSV containing more than 34,000 organizations that have received funding from the charitable organization over the years was used. Within this dataset, there are a number of columns that capture metadata about each organization.

* Objective 1 - Preprocessing Data for a Neural Network Model

* Objective 2 - Compile, Train, and Evaluate the Model

* Objective 3 - Optimize the Model

* Objective 4 - Summarize the performance of the Neural Network Model in a written report (README.md)


#### Project Resources: 
Python, Pandas, Scikit-Learn

## Results:

### Data Preprocessing

#### Target variable in machine learning

A target(y) variable in machine learning is the final output you are trying to predict. It can be categorical (alive vs. dead) or continuous (price of a horse).

The variable considered the target of the model was: 

* IS_SUCCESSFUL

#### Feature variable(s) in machine learning

A feature(or column) in machine learning represents a measurable piece of data that can be used for analysis Features are sometimes also referred to as variables. 
The variables considered to be features were: 

* APPLICATION_TYPE — Alphabet Soup application type
* AFFILIATION — Affiliated sector of industry
* CLASSIFICATION — Government organization classification
* USE_CASE — Use case for funding
* ORGANIZATION — Organization type
* STATUS — Active status
* INCOME_AMT — Income classification
* SPECIAL_CONSIDERATIONS — Special consideration for application
* ASK_AMT — Funding amount requested

The variables consider neither targets nor features that were removed from the input data (at least initially):

* EIN — Identification columns
* NAME — Identification columns

### Compiling, Training, and Evaluating the Model
* Initially the deep learning model did not include the NAME data, bundled APPLICATION_TYPE into an "Other" category where APPLICATION_TYPE counts were < 500, bundled CLASSIFICATION into an "Other" category where CLASSIFICATION counts  were < 500, had 2 hidden layers with 75 neurons in the first layer and 30 in the second, ran with 25 epochs and achieved a testing accuracy of 73%.
* The first attempt at optimization added the NAME data and bundled that data into an "Other" category where NAME counts  were < 300, bundled APPLICATION_TYPE into an "Other" category where APPLICATION_TYPE counts were < 500, bundled CLASSIFICATION into an "Other" category where CLASSIFICATION counts  were < 500, kept the same 2 hidden layers with 30 neurons in the first layer and 10 in the second hidden layer, ran with 25 epochs achieving an accuracy of 74%.
* The second attempt at optimization added the NAME data and bundled that data into an "Other" category where NAME counts  were < 200, bundled APPLICATION_TYPE into an "Other" category where APPLICATION_TYPE counts were < 500, bundled CLASSIFICATION into an "Other" category where CLASSIFICATION counts  were < 500, kept the same 2 hidden layers with 30 neurons in the first layer and 10 in the second hidden layer achieving an accuracy of 75%.
* The third attempt at optimization added the NAME data and bundled that data into an "Other" category where NAME counts  were < 50, bundled APPLICATION_TYPE into an "Other" category where APPLICATION_TYPE counts were < 50, bundled CLASSIFICATION into an "Other" category where CLASSIFICATION counts  were < 50, kept the same 2 hidden layers and changed to 40 neurons in the first layer and 5 in the second hidden layer, ran with 25 epochs achieving an accuracy of 76%.

## Evaluating the last optimization attempt

The analysis was able to achieve the target model performance of 76%.

* Bringing NAME into the analysis, encoding it, and adjusting the bundling of the data increased model performance.

Initial deep learning model results:

![initial model accuracy image](/resources/initial_model_accuracy.png)


![initial model graph image](/resources/initial_model_accuracy_loss_graph.png)


Optimization attempt 1 deep learning model results:

![Optimization attempt 1 accuracy image](/resources/opt1_accuracy.png)


![Optimization attempt 1 graph image](/resources/opt1_model_accuracy_loss_graph.png)


Optimization attempt 2 deep learning model results:

![Optimization attempt 2 accuracy image](/resources/opt2_accuracy.png)


![Optimization attempt 2 graph image](/resources/opt2_model_accuracy_loss_graph.png)


Optimization attempt 3 deep learning model results:

![Optimization attempt 3 accuracy image](/resources/opt3_accuracy.png)


![Optimization attempt 3 graph image](/resources/opt3_model_accuracy_loss_graph.png)


## Summary:

The overall results of the deep learning model optimzation were to improve the model from 73% to 76%. 

Most of the gains were made in changes to the data in data preprocessing not by changing models or changes to layer configurations. 

### Recommendations:

Recommendations for how a different model could solve this classification problem include using a Logistic Regression model or any Decision Tree model. 

Either of these would allow for an easier time explaining how the results were achieved and may well achieve similar accuracy levels.

### References:
* Beginners Ask "How Many Hidden Layers/Neurons to Use in Artifical Neural Networks?" https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e
* Display Deep Learning Model Training History in Keras: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
* The neural network playground: https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.81307&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&discretize_hide=true&regularization_hide=true&learningRate_hide=true&regularizationRate_hide=true&percTrainData_hide=true&showTestData_hide=true&noise_hide=true&batchSize_hide=true