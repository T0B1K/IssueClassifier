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
3. run [train.py] (classifier/train.py). Make changes in the loadConfig.json to fit the training to your needs and change the classifiers / labels in train.py to train the specific classifier.

## instructions for running the crawler
Just open the [crawler_and_analysis_tool](github_crawler/crawler_and_analysis_tool.html) and run the file.
and paste in the information required. The crawling status can be seen at the top of the page, after clicking on the "start" button.
To sanity check the issues crawled, just click on "prev" or "next" to change the current page

The crawler is also used as analysis tool for sanity checking after the issues have been downloaded, to use it as such, open the [github_crawler/crawler_and_analysis_tool.html](github_crawler/crawler_and_analysis_tool), click on "browse" and open the specific .json file.
To sanity check the issues crawled, just click on "prev" or "next" to change the current page 

## directory structure

### classifier:
  Contains the current state of development. It includes the trained classifiers and the logic of how the classifiers work together to add an issue to its related class.
### classifier_doku:
   [This](classifier_doku/index.html) contains all the html documentation pages for the python files in [classifier](classifier)
### github_crawler:
  This folder contains the HTML file and the related scripts/stylesheets that allow you to crawl and analyze issues from GitHub repositories.

### issues:
  This folder contains all crawled issues so far. "To add" contains all issues not jet added to our dataset.

### microservice:
  This folder contains all the dokuments required to run the microservice

### results:
  This contains all output files from the software that let us review the quality of the software. 
