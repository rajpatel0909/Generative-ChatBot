import tensorflow as tf
import pandas as pd
import numpy as np
import re
import time


print("preprocessing...")
movieL = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
movieC = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

idToLine = {}
for l in movieL:
    w = l.split(' +++$+++ ')
    if len(w) == 5:
        idToLine[w[0]] = w[4]

conversations = [ ]
for l in movieC[:-1]:
    w = l.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations.append(w.split(','))

context = []
responses = []

for conversation in conversations:
    for i in range(len(conversation)-1):
        context.append(idToLine[conversation[i]])
        responses.append(idToLine[conversation[i+1]])


replacements = open('replacements.txt', 'r').read().split('\n')

def replaceText(txt):

    txt = txt.lower()
    
    for replace in replacements:
        replacement = replace.split(',')
        txt = re.sub(replacement[0], replacement[1], txt)
    
    txt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", txt)
    return txt

replacedQues = []
for que in context:
    replacedQues.append(replaceText(que))
    
replacedAns = []    
for ans in responses:
    replacedAns.append(replaceText(ans))

queTemp = []
ansTemp = []

i = 0
for que in replacedQues:
    if len(que.split()) >= 2 and len(que.split()) <= 20:
        queTemp.append(que)
        ansTemp.append(replacedAns[i])
    i += 1

smallQue = []
smallAns = []

i = 0
for ans in ansTemp:
    if len(ans.split()) >= 2 and len(ans.split()) <= 20:
        smallAns.append(ans)
        smallQue.append(queTemp[i])
    i += 1

vocabulary = {}
for que in smallQue:
    for w in que.split():
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1
            
for ans in smallAns:
    for w in ans.split():
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1


count = 0
for k,v in vocabulary.items():
    if v >= 10:
        count += 1

quev2int = {}

word_num = 0
for word, count in vocabulary.items():
    if count >= 10:
        quev2int[word] = word_num
        word_num += 1
        
ansv2int = {}

word_num = 0
for word, count in vocabulary.items():
    if count >= 10:
        ansv2int[word] = word_num
        word_num += 1

wilds = ['<PAD>','<EOS>','<UNK>','<GO>']

for wild in wilds:
    quev2int[wild] = len(quev2int)+1
    
for wild in wilds:
    ansv2int[wild] = len(ansv2int)+1

queint2v = {v_i: v for v, v_i in quev2int.items()}
ansint2v = {v_i: v for v, v_i in ansv2int.items()}

for i in range(len(smallAns)):
    smallAns[i] += ' <EOS>'

queInt = []
for que in smallQue:
    ints = []
    for w in que.split():
        if w not in quev2int:
            ints.append(quev2int['<UNK>'])
        else:
            ints.append(quev2int[w])
    queInt.append(ints)
    
ansInt = []
for ans in smallAns:
    ints = []
    for w in ans.split():
        if w not in ansv2int:
            ints.append(ansv2int['<UNK>'])
        else:
            ints.append(ansv2int[w])
    ansInt.append(ints)
    
    
train_valid_split = int(len(queInt)*0.15)

train_questions = queInt[train_valid_split:]
train_answers = ansInt[train_valid_split:]

valid_questions = queInt[:train_valid_split]
valid_answers = ansInt[:train_valid_split]

