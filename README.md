# misclass

**misclass** is a tool for understanding fastText classification model prediction errors.

Often when we train and test a model, we only look at the accuracy.  Seeing actual examples of misclassified rows, and the specific segments 

This script can be used for diagnosting reasons of misclassyifing data with fasttext. You can train your model with fasttext and pass it to this tool. After that you can see misclassified test data with word n-grams. This will help you to understand reasons of misclassifying. You are free to choose n-gram size and steps.

### Example

Consider a model trained to do a simple classification task like sentiment analysis.  The model makes an error, it incorrectly predicts the following Amazon review is `negative` when the correct label `positive`. 

> I bought this for my friend who plays the piano.  Honestly I was not expecting much because of some bad experiences with these types of products in the past.  In the end though this one was definitely worth the money and I even ended up buying another one!

misclass can highlight the segments whose predicted label did not match the correct label and thus caused the prediction for the whole row to fail.

> I bought this for my friend who plays the piano.  Honestly I was **not expecting much** because of some **bad experiences** with these types of products in the past.  In the end though this one was definitely worth the money and I even ended up buying another one!

misclass does this by iterating through each segment and checking it against the label.  fastText predictions are calculated by averaging the predictions across all segments.

## Parameters
There are parameters that control the way that the row is divided into segments for highlighting.

    `step`: 

    `countOfWords`: 


## Running

The full set of parameters required by misclass.py is:

    `testPath`: Path to the test data.
  
    `modelPath`: Path to the trained model to evaluate.
  
    `output`: Path to which to write a file with the misclass output
  
    `countOfWords`: Count of words in test sentence.
  
    `step`: Step for taking words in test sentence.
  
For example:

```
python misclass.py --testPath test.txt --modelPath model --output misclass.txt --countOfWords 5 --step 3
```

