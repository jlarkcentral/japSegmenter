#!/usr/bin/env python3.2
# -*- coding: utf-8 -*-

'''

japanese segmentation

usage:

python3.2 mySegmenter.py <trainfile> <testfile>

'''


import sys
import xml.etree.ElementTree as ET
import re
import math
import xml.sax
import codecs
import operator
from collections import defaultdict


'''
# test on ONE char
def test(model,filenameTest):
	foundSentences = []
	tree = ET.parse(filenameTest)
	root = tree.getroot()
	for s in root.findall('sentence'):
		rawtext = s.find('raw').text
		cstate = 'B'
		myToken = ''
		mySentence = []
		for i in range(0, len(rawtext)):
			c = rawtext[i]
			if (mostProbable(model,cstate,c)) == 'B':
				myToken += c
				mySentence.append(myToken)
				myToken = ''
				cstate = 'B'
			elif (mostProbable(model,cstate,c)) == 'C':
				myToken += c
				cstate = 'C'
		foundSentences.append(mySentence)

	handle = codecs.open("grotest", 'w', 'utf-8')
	handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
	handle.write('<dataset>\n')
	for i in range(len(foundSentences)):
	    segmented_sentence = " ".join(foundSentences[i])
	    handle.write('\t<sentence sid="'+str(i)+'">\n')
	    handle.write('\t\t<raw>'+segmented_sentence+'</raw>\n')
	    handle.write('\t</sentence>\n')
	handle.write('</dataset>')
	handle.close()
'''

'''
# observations on ONE char aat
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
'''





# load [raw,segmentation indices]
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




# observations on bigrams
def train(sentences):
	obs = defaultdict(lambda: defaultdict(lambda: 1))
	tr = defaultdict(lambda: defaultdict(lambda: 0))
	nb = 0
	for raw,indices in sentences:
		j = 0
		i = 0
		cstate = 'B'
		while i < len(raw) - 1:
			bigram = raw[i]+raw[i+1]
			if indices[j] == i+1:
				obs['B'][bigram] += 1
				j += 1
				tr[cstate]['B'] += 1
				cstate = 'B'
			else:
				obs['C'][bigram] += 1
				tr[cstate]['C'] += 1
				cstate = 'C'
			i += 1
			nb += 1
	#for a in tr:
	#	for b in tr[a]:
	#		tmp = float(tr[a][b]/nb)
	#		tr[a][b] = tmp 
	return list([obs,tr])




# most probable state for this bigram observation
def mostProbable(model,cstate,bigram):
	otherState = 'B' if cstate == 'C' else 'C' 
	d = False
	if not bigram in model[0]['B'] or not bigram in model[0]['C']:
		#print("NOT in dict")
		cPb = 1
		for bg in model[0][cstate]:
			if bigram[0] in bg or bigram[1] in bg:
				cPb += 1
		oPb = 1
		for bg in model[0][otherState]:
			if bigram[0] in bg or bigram[1] in bg:
				oPb += 1

		cstatePb = model[1][cstate][cstate] * cPb
		otherStatePb = model[1][cstate][otherState] * oPb
		if abs((float(min(cstatePb,otherStatePb)) / max(cstatePb,otherStatePb))) < 0.3:
			d = True
	
	if not d:
		#print("IN dict") 
		cstatePb = model[1][cstate][cstate] * model[0][cstate][bigram]
		otherStatePb = model[1][cstate][otherState] * model[0][otherState][bigram]
	
	maxState = cstate if cstatePb > otherStatePb else otherState
	return maxState




def test(model,filenameTest):
	foundSentences = []
	tree = ET.parse(filenameTest)
	root = tree.getroot()
	for s in root.findall('sentence'):
		rawtext = s.find('raw').text
		cstate = 'B'
		myToken = ''
		indices = []
		i = 0
		while i < len(rawtext) - 1:
			bg = rawtext[i]+rawtext[i+1]
			if (mostProbable(model,cstate,bg)) == 'B':
				indices.append(i+1)
			#elif (mostProbable(model,cstate,bg)) == 'C':

			i += 1
		indices.append(len(rawtext))
		mySentence = sentencizer(rawtext,indices)
		foundSentences.append(mySentence)

	handle = codecs.open("grotest.xml", 'w', 'utf-8')
	handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
	handle.write('<dataset>\n')
	for i in range(len(foundSentences)):
	    segmented_sentence = " ".join(foundSentences[i])
	    handle.write('\t<sentence sid="'+str(i)+'">\n')
	    handle.write('\t\t<raw>'+segmented_sentence+'</raw>\n')
	    handle.write('\t</sentence>\n')
	handle.write('</dataset>')
	handle.close()




def sentencizer(raw,indices):
	sentence = []
	ctok = ''
	j = 0
	for i in range(len(raw)):
		if indices[j] == i:
			sentence.append(ctok)
			ctok = ''
			j += 1
		ctok += raw[i]
		if i == len(raw) - 1:
			sentence.append(ctok)
	return sentence



##############################################################################

def main():
	sentences = loadTrainSentences(sys.argv[1])
	model = train(sentences)
	test(model,sys.argv[2])





if __name__ == '__main__':
    sys.exit(main())