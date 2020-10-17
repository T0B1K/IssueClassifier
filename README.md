# Issue Classifier

## instructinos for running the microservice
please refer to the [microservice dokumentation](microservice/README.md) for further information

## instructions for creating classifiers
1. Make sure you have at least Python 3.7 installed.
2. Install all the necessary libraries.
   - numpy
   - pandas
   - seaborn
   - joblib
   - matplotlib.pyplot
   - sklearn
   - nltk
3. run [classifier/train.py]. Make changes in the loadConfig.json to fit the training to your needs and change the classifiers / labels in train.py to train the specific classifier.

## directory structure

### classifier:
  Contains the current state of development. It includes the trained classifiers and the logic of how the classifiers work together to add an issue to its related class.

### github_crawler:
  This folder contains the HTML file and the related scripts/stylesheets that allow you to crawl and analyze issues from GitHub repositories.

### issues:
  This folder contains all crawled issues so far. To add contains all issues not     added to our dataset.

### microservice:
    

### results:
  This contains all output files from the software that let us review the quality of the software. 
