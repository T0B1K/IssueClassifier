# Automatic issue classifier
## Introduction
In the context of software development projects, issues provide a vital tool to describe a variety of tasks to be accomplished by the team. Four of the most common types of issues are *feature requests*, *bug reports*, and tasks related to *documentation*, as well as tasks related to an *api*. The state-of-the-art involves manual classification of issues into their respective categories. While this requires marginal effort for projects of minimal scale with teams of comparably minimal size, large-scale projects involving multiple teams from different organisations pose a much larger problem in that regard. Mislabelling issues can lead to subsequent erroneous prioritisation of issues, resulting in misplaced time and energy. Furthermore, issues left unlabelled make it harder for project managers to pinpoint specific sources of errors and bugs in the software, whose impact can range from trivial to severe under certain circumstances.
## Problem statement
These problems can be summarised into one problem statement: Software development teams require accurate classification of every software project issue promptly, calling for the need of an automatic issue classifier. Several attempts have been made to curb variants of this problem, for example by developing a GitHub app to automatically classify issues. However, several key aspects are missing from them: None of them can be easily integrated into software projects spanning multiple components from separate teams using different issue management systems.
## Our solution
We are providing a possible solution by deploying an automatic issue classifier in form of a microservice which classifies issues based on their body texts, returning the suggested label(s) most appropriate for the issue.
>For example, a bug related to an API of a component could be labelled as both "bug" and "api", while additions to documentation "docu" and "feature request".

## How we addressed the issue
### Crawler
First we created a github [crawler](github_crawler/) which automatially crawls the issues from manually selected github repositories and saves them in into *.json* files.
> I.e. one crawls the bugs from the repository *demoRepo* made by person *MrSmith*, it will be saved as *MrSmith_demoRepo_bug.json* and the issues crawled will look like
> ```json 
> [{"labels":["bug"],"text":"Houston we have a problem"},{"labels":["bug"],"text":"..."},...]
Using the GitHub crawler those issues can also be inspected to check whether or not they make sense and further adjustments can be made - refer to the [crawler documentation](github_crawler/) **[TODO]** for more informations regarding the crawler and sanity checking.

### Issue classifier
After having crawled multiple issues we began creating issue classifiers and training them by using the crawled issues and issue- labels. \
But before classifing them, they have to be vectorized.
#### Vectorizer
We trained an [tfidf vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidf#sklearn.feature_extraction.text.TfidfVectorizer) by giving it data to vectorized. We used the following adjustments
adjustment|why we made that adjustment
---|---
ngram: tuple = (1, 2) | ngrams are used to see the word in the context of their neighbors - it was decided against larger ngrams due to the space complexity
stripAccents=None | stripping non unicode caraters or not didn't make a whole lot of difference, because we used just english 
stopWords=None |...  
**[TODO] word occurences > 3**
#### Classifier
We are using following estimators provided by sykit-learn.
estimators | modifications |
----|---
[MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html?highlight=multinomialnb#sklearn.naive_bayes.MultinomialNB)| -
[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgdclassifier#sklearn.linear_model.SGDClassifier)| (loss='modified_huber', penalty='l2',alpha=1e-3, random_state=100, max_iter=200)
[sigmoidSVM](https://scikit-learn.org/stable/modules/classes.html?highlight=svm#module-sklearn.svm) | SVC(kernel='sigmoid', gamma=1.0))
[RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier)| RandomForestClassifier(200, bootstrap=False))
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression)| (solver='sag',random_state=100)

One can also take different classifiers, add or delete them.
Using those classifiers, each classifier decides for itself whether or not an issue is i.e. a bug or an enhancement. After each classifier decided what the issue describes, another classifier classifies their results and guesses the right answer (This method is called stacking, because one is stacking classifiers).
During our tests we found out, that a normal democratic majority vode outperformes this kind of stacking by about 1 Percent. Therefore we are letting the user decide, which kind they want to use. **[TODO]**
        