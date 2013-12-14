#!/usr/bin/env python3.2
# -*- coding: utf-8 -*-

'''

naive japanese segmentation

usage:
python mySegmenter.py <trainfile> <testfile>

'''


import sys
import xml.etree.ElementTree as ET
import re
import math
import xml.sax
import codecs
import operator
from collections import defaultdict
from random import choice



def loadTrainSentences(filenameTrain):
	sentences = []
	tree = ET.parse(filenameTrain)
	root = tree.getroot()
	for s in root.findall('sentence'):
		raw = s.find('raw').text
		indices = []
		ts = s.find('tokens')
		i = 0
		for t in ts.findall('token'):
			i += len(t.text)
			indices.append(i)
		sentences.append([raw,indices])
	return sentences





def train(sentences):
	obs = defaultdict(lambda: defaultdict(lambda: 0))
	tr = {}
	for raw,indices in sentences:
		j = 0
		i = 0
		while i < len(raw) - 1:
			w1 = raw[i]
			if indices[j] == i+1:
				obs[w1]['B'] += 1
				j += 1
			else:
				obs[w1]['C'] += 1
			i += 1
	return obs



def mostProbable(model,char):
	if not char in model:
		return 'B'
	else:
		out = ['B','C']
		values = [model[char]['B'],model[char]['C']]
		i = values.index(max(values))
		return out[i]



def test(model,filenameTest):
	foundSentences = []
	tree = ET.parse(filenameTest)
	root = tree.getroot()
	seen = []
	for s in root.findall('sentence'):
		rawtext = s.find('raw').text
		myToken = ''
		mySentence = []
		for i in range(0, len(rawtext)):
			c = rawtext[i]
			if (mostProbable(model,c)) == 'B':
				myToken += c
				mySentence.append(myToken)
				myToken = ''
			elif (mostProbable(model,c)) == 'C':
				myToken += c
		foundSentences.append(mySentence)
	handle = codecs.open(sys.argv[3], 'w', 'utf-8')
	handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
	handle.write('<dataset>\n')
	for i in range(len(foundSentences)):
	    segmented_sentence = " ".join(foundSentences[i])
	    handle.write('\t<sentence sid="'+str(i)+'">\n')
	    handle.write('\t\t<raw>'+segmented_sentence+'</raw>\n')
	    handle.write('\t</sentence>\n')
	handle.write('</dataset>')
	handle.close()






##############################################################################

def main():
	if len(sys.argv) != 4:
		print("usage:\npython mySegmenter_naive.py <trainfile> <testfile> <output>")
	else:
		sentences = loadTrainSentences(sys.argv[1])
		model = train(sentences)
		test(model,sys.argv[2])


if __name__ == '__main__':
    sys.exit(main())