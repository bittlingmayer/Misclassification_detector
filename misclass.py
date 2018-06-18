import sys
import re
import subprocess
import shlex
import numpy as np
import argparse

tmpOutput = open('tmp.txt', 'w')

def GetNgrams(text, countOfWords, step):
	text = text.strip()
	splittedText = text.split(' ')
	space = ' '
	ngrams = [space.join(splittedText[i : i + countOfWords]) for i in range(0, len(splittedText) - 1, step)]
	return ngrams

def GetMisclassifiedData(labeledData, modelPath, testPath):
	labels = re.findall('__label__[1-9]{1,}', labeledData)
	rows = filter(None, re.split('__label__[1-9]{1,}', labeledData))

	command = shlex.split("../fastText/fastText-0.1.0/./fasttext predict-prob " + modelPath  + " " + testPath)

	process = subprocess.Popen(command, stdout=subprocess.PIPE)
	output, err = process.communicate()

	receivedLabels = re.findall('__label__[1-9]{1,}', output)
	print len(labels), len(receivedLabels)

	labels = np.array(labels)
	receivedLabels = np.array(receivedLabels)
	
	differences = [i != j for i, j in zip(labels, receivedLabels)]
	differences = np.array(differences, dtype = np.uint8)
	return [rows[i] for i in np.nonzero(differences)[0]], [labels[i] for i in np.nonzero(differences)[0]]
	
def AddMisclassifiedNGrams(data, labels, countOfWords, step):
	global tmpOutput
	for i in range(len(data)):
		tmpOutput.write(str(labels[i]) + data[i].strip() + '\n')
		nGrams = GetNgrams(data[i].strip(), countOfWords, step)
		for nGram in nGrams:
			tmpOutput.write(str(labels[i]) + nGram + '\n')		
	tmpOutput.close()
	return


parser = argparse.ArgumentParser()

parser.add_argument('--testPath', help = 'Path of test data')
parser.add_argument('--modelPath', help = 'Path of trained model')
parser.add_argument('--output', help = 'Output file')
parser.add_argument('--countOfWords', help = 'Count of words in test sentence')
parser.add_argument('--step', help = 'Step for taking words in test sentence')

parsedArgs = parser.parse_args()

testPath = parsedArgs.testPath
modelPath = parsedArgs.modelPath
output = parsedArgs.output
countOfWords = int(parsedArgs.countOfWords)
step = int(parsedArgs.step)

outputFile = open(output, 'w')

testFile = open(testPath, 'r')
testData = testFile.read()

wrongDataRows, wrongDataLabels = GetMisclassifiedData(testData, modelPath, testPath)

AddMisclassifiedNGrams(wrongDataRows, wrongDataLabels, countOfWords, step)

tmp = open('tmp.txt', 'r')
tmp = tmp.read()
result, resultLabels = GetMisclassifiedData(tmp, modelPath, 'tmp.txt')

for res, label in zip(result, resultLabels):
	outputFile.write(str(label) + res)

outputFile.close()

