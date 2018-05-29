# Misclassification_detector

This is a tool for getting misclassified data for fasttext. 

## Overview
    
This script can be used for diagnosting reasons of misclassyifing data with fasttext. You can train your model with fasttext and pass it to this tool. After that you can see misclassified test data with word n-grams. This will help you to understand reasons of misclassifying. You are free to choose n-gram size and steps.

There are 2 parameters for customizing  diagnosting 

```
--step
```

and

```
--countOfWords
```
   
    
## Parameters

  --testPath: Path of test data.
  
  --modelPath: Path of trained model.
  
  --output: Path of output file where will be stored result of script.
  
  --countOfWords: Count of words in test sentence.
  
  --step: Step for taking words in test sentence.
