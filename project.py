import sys
import re
import subprocess
import shlex
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--testPath', help = 'Path of test data')
parser.add_argument('--modelPath', help = 'Path of trained model')
parser.add_argument('--output', help = 'Output file')

parsedArgs = parser.parse_args()

testPath = parsedArgs.testPath
modelPath = parsedArgs.modelPath
output = parsedArgs.output

outputFile = open(output, 'w')

testFile = open(testPath, 'r')
testData = testFile.read()

labels = re.findall('__label__[1-9]{1,}', testData)
rows = filter(None, re.split('__label__[1-9]{1,}', testData))

command = shlex.split("../fastText/fastText-0.1.0/./fasttext predict-prob " + modelPath  + " " + testPath)

process = subprocess.Popen(command, stdout=subprocess.PIPE)
output, err = process.communicate()

receivedLabels = re.findall('__label__[1-9]{1,}', output)

labels = np.array(labels)
receivedLabels = np.array(receivedLabels)

differences = [i != j for i, j in zip(labels, receivedLabels)]
differences = np.array(differences, dtype = np.uint8)
#print np.count_nonzero(differences)
#print float(len(differences) - np.count_nonzero(differences))/len(differences)

#print np.nonzero(differences)[0]
for i in np.nonzero(differences)[0]:
	#print rows[i]
	outputFile.write(rows[i].strip() + '\n' + '\n')

#print output
outputFile.close()
#print output

