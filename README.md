# Filtering article into category by using LSTM

## Project Description
Text documents are essential as they are one of the richest sources of data for businesses. Text documents often contain crucial information which might shape the market trends or influence the investment flows. Therefore, companies often hire analysts to monitor the trend via articles posted online, tweets on social media platforms such as Twitter or articles from newspaper. However, some companies may wish to only focus on articles related to technologies and politics. Thus, filtering of the articles into different categories is required.

Thus, this project came to light by creating a LSTM model to categorize unseen article into 5 categories namely Sport,Tech,Business,Entertainment and Politics.


# How to install and run the project
1)Click on this provided link, you will be redirected to [GoogleColab](https://colab.research.google.com/drive/1vRbadX2pLrHV5ooFUh8UfObuQFFvjPzr?usp=sharing),
you may need to sign in to your Google account to access it.

2)You also can run the model by using spyder. By downloading all the folder and files insde the repository, you can open the latest version of spyder and running the multiclass_nlp.py to run the model. Ensure that multiclass_nlp_module.py are in the same path as multiclass_nlp.py.

For pc user: 
Software required: Spyder, Python(preferably the latest version) 
Modules needed: Tensorflow, Sklearn

# Model Architecture

![PlotModel](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/9bd895f7cbed6fa26e4236ec979baeda7f25f858/static/model.png)

# Execution

## Training Loss

![PlotLoss](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/9bd895f7cbed6fa26e4236ec979baeda7f25f858/static/hist_plot_loss.png)

## Training Accuracy

![PlotLoss](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/9bd895f7cbed6fa26e4236ec979baeda7f25f858/static/hist_plot_acc.png)

# Tensorboard Result

## Tensorboard Loss Graph

![TensorLoss](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/9bd895f7cbed6fa26e4236ec979baeda7f25f858/static/tensorboard_loss.PNG)

## Tensorboard Acc Graph

![TensorAcc](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/9bd895f7cbed6fa26e4236ec979baeda7f25f858/static/tensorboard_acc.PNG)

# Project Outcome
As attached below the result of the model f1 score obtained are 0.93 accuracy.

![F1Score](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/9bd895f7cbed6fa26e4236ec979baeda7f25f858/static/F1_score.PNG)


![Matrix](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/9bd895f7cbed6fa26e4236ec979baeda7f25f858/static/Confusion_matrix_display.png)

# Discussion 
As shown graph below, it show that the data are imbalance, by balancing the data by adding the data make it 500 or reduce all the data into 400 will be able to improve the model accuracy.

![target](https://github.com/shahirilfauzan/Multiclass-Article-Classification/blob/8babee4b0a203c4765c518db49805305bf5641b2/static/target_graph.png)

# Credits
this dataset is provided by [Susan Li](https://github.com/susanli2016),[Dataset](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv)

# This project running with

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
