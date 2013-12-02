#!/usr/bin/env python3.2
# -*- coding: utf-8 -*-

'''
japanese segmentation

usage:
python mySegmenter.py <trainfile> <testfile>

author: j. lark
'''

import sys
import xml.etree.ElementTree as ET
import re
import math
import xml.sax
import codecs
import operator
from collections import defaultdict
from progressbar import ProgressBar,Percentage,Bar


# load sentences with tokens boundaries
def loadTrainSentences(filenameTrain):
	print("Loading train sentences...")
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
	print("Training the model...")
	obs = defaultdict(lambda: defaultdict(lambda: 1))
	prevObs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1)))
	tr = defaultdict(lambda: defaultdict(lambda: 0))

	prevBigram = "start"
	for raw,indices in sentences:
		j = 0
		i = 0
		cstate = 'B'
		while i < len(raw) - 1:
			bigram = raw[i:i+2]
			if indices[j] == i+1:
				obs['B'][bigram] += 1
				j += 1
				tr[cstate]['B'] += 1
				cstate = 'B'
				prevObs['B'][bigram][prevBigram] += 1
			else:
				obs['C'][bigram] += 1
				tr[cstate]['C'] += 1
				cstate = 'C'
				prevObs['C'][bigram][prevBigram] += 1
			i += 1
			prevBigram = bigram

	return [obs,tr,prevObs]


# identify tokens on test sentences
def test(model,filenameTest):
	print("Running the test...")
	foundSentences = []
	tree = ET.parse(filenameTest)
	root = tree.getroot()
	pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(root.findall('sentence'))).start()
	k = 0
	pvBg = "start"
	for s in root.findall('sentence'):
		rawtext = s.find('raw').text
		bigrams = []
		i = 0
		while i < len(rawtext) - 1:
			bg = rawtext[i:i+2]
			bigrams.append(bg)
			i += 1
		path = mostProbablePath(model,bigrams)
		indices = indicesFromPath(path)
		foundSentences.append(sentencizer(rawtext,indices))
		k += 1
		pbar.update(k)
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


# most probable sequences of states for this bigram observation
def mostProbablePath(model,bigrams):
	path = []
	probas = {}
	pvBg = "start"
	for bg in bigrams:
		if path == []:
			probas = dict({'B':model[0]['B'][bg], 'C':model[0]['C'][bg]})
		nextP = {}
		for s in ['B','C']:
			nextProb = nextProbas(model,s,bg,pvBg)
			proba = max(probas['B'] * nextProb[s],probas['C'] * nextProb[s])
			nextP[s] = proba
		path.append(max(nextP, key=nextP.get))
		probas = nextP
		pvBg = bg
	return path


# next probabilities for a given state and bigram
def nextProbas(model,cstate,bigram,prevBigram):
	observations = model[0]
	transitions = model[1]
	prevObservations = model[2]
	d = False
	'''
	if not bigram in observations['B'] or not bigram in observations['C']:
		bCoeff = 1
		for bg in observations['B']:
			if bigram[0] in bg or bigram[1] in bg:
				bCoeff += 1
		cCoeff = 1
		for bg in observations['C']:
			if bigram[0] in bg or bigram[1] in bg:
				cCoeff += 1
		bPb = prevObservations['B'][prevBigram][bigram] * transitions[cstate]['B'] * bCoeff
		cPb = prevObservations['C'][prevBigram][bigram] * transitions[cstate]['C'] * cCoeff
		if abs((float(min(bPb,cPb)) / max(bPb,cPb))) < 0.3:
			d = True
	'''
	if not d:
		bPb = transitions[cstate]['B'] * observations['B'][bigram]
		cPb = transitions[cstate]['C'] * observations['C'][bigram]

	return dict({'B':bPb,'C':cPb})


# get boundaries from sequence fo states
def indicesFromPath(path):
	indices = []
	for i in range(len(path)):
		if path[i] == 'B':
			indices.append(i+1)
	if path[-1] == 'C':
		indices.append(len(path))
	return indices


# get sentence from tokens boundaries
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



######################
#		Main         #
######################

def main():
	sentences = loadTrainSentences(sys.argv[1])
	model = train(sentences)
	test(model,sys.argv[2])


if __name__ == '__main__':
    sys.exit(main())