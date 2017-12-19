import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

contexts = []
utterances = []
targets = []
cntr = 0
filename = 'data/train.csv'
file = open(filename, encoding="utf8")
csvReader = csv.reader(file)
limit  = 10000
for row in csvReader:
    contexts.append(row[0])
    utterances.append(row[1])
    targets.append(row[2])
    cntr += 1
    if cntr == limit:
        break

targets = np.reshape(targets, (len(targets),1))

replacements = open('replacements.txt', 'r').read().split('\n')
def replaceText(txt):

    txt = txt.lower()
    
    for replace in replacements:
        replacement = replace.split(',')
        txt = re.sub(replacement[0], replacement[1], txt)
    
    txt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,_]", "", txt)
    txt = re.sub(r"eou", "", txt)
    txt = re.sub(r"eot", "", txt)
    return txt

replacedQues = []
replacedAns = []    

print('initially')
for i in range(2):
    print(contexts[i])
    print(utterances[i])

for que in contexts:
    replacedQues.append(replaceText(que))
     
for ans in utterances:
    replacedAns.append(replaceText(ans))

print('replaced')
for i in range(2):
    print(replacedQues[i])
    print(replacedAns[i])

# removing conversations with large difference in lengths
consiseQues = replacedQues
consiseAns = replacedAns

vocabulary = {}
for que in contexts:
    for w in que.split():
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1
             
for ans in utterances:
    for w in ans.split():
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1
 
print (len(vocabulary))



queW2int = {}
cntr = 4
for cont in consiseQues:
    for w in cont.split():
        if w not in queW2int:
            queW2int[w] = cntr;
            cntr += 1
            
ansW2int = {}
cntr = 4
for cont in consiseAns:
    for w in cont.split():
        if w not in ansW2int:
            ansW2int[w] = cntr;
            cntr += 1

wilds = ['<PAD>','<EOS>','<UNK>','<GO>']

cntr = 0
for wild in wilds:
    queW2int[wild] = cntr
    cntr += 1
    
cntr = 0 
for wild in wilds:
    ansW2int[wild] = cntr
    cntr += 1
    
queint2W = {v_i: v for v, v_i in queW2int.items()}
ansint2W = {v_i: v for v, v_i in ansW2int.items()}



queInt = []
for que in consiseQues:
    ints = []
    for w in que.split():
        if w not in queW2int:
            ints.append(queW2int['<UNK>'])
        else:
            ints.append(queW2int[w])
    queInt.append(ints)
    
ansInt = []
for ans in consiseAns:
    ints = []
    for w in ans.split():
        if w not in ansW2int:
            ints.append(ansW2int['<UNK>'])
        else:
            ints.append(ansW2int[w])
    ansInt.append(ints)
    

print('len of queInt', len(queInt))
print('len of andInt', len(ansInt))
print('len of targets', len(targets))

train_valid_split = int(len(queInt)*0.15)

train_questions = queInt[train_valid_split:]
train_answers = ansInt[train_valid_split:]
train_targets = targets[train_valid_split:]

valid_questions = queInt[:train_valid_split]
valid_answers = ansInt[:train_valid_split]
valid_targets = targets[:train_valid_split]

PAD = queW2int['<PAD>']
for i in range(0, len(train_questions)):
    ans_len = len(train_answers[i])
    que_len = len(train_questions[i])
    if que_len > 50:
        train_questions[i] = train_questions[i][:50]
    else:
        extra_array_que = [PAD] * (50 - (que_len))
        train_questions[i] += extra_array_que
        
    if ans_len > 50:
        train_answers[i] = train_answers[i][:50]
    else:
        extra_array_ans = [PAD] * (50 - (ans_len))
        train_answers[i] += extra_array_ans
        
    if i < 25:
        print (train_answers[i])
        print(len(train_answers[i]))
        print (train_questions[i])
        print(len(train_questions[i]))
        
for i in range(0, len(valid_questions)):
    ans_len = len(valid_answers[i])
    que_len = len(valid_questions[i])
    extra_array = [PAD] * abs(ans_len - que_len)
    if ans_len > que_len:
        valid_questions[i] += extra_array
    else:
        valid_answers[i] += extra_array
        

print (len(train_questions))

for i in range(5):
    print(i, queint2W[i], "->", queW2int[queint2W[i]])
    
for i in range(5):
    print(i, ansint2W[i], "->", ansW2int[ansint2W[i]])

#store train_questions train_answers valid_questions and valid_answers and word2int and int2word
data = {}
data['train_questions'] = train_questions
data['train_answers'] = train_answers
data['train_targets'] = train_targets
data['valid_questions'] = valid_questions
data['valid_answers'] = valid_answers
data['valid_targets'] = valid_targets
data['queW2int'] = queW2int
data['queint2W'] = queint2W
data['ansW2int'] = ansW2int
data['ansint2W'] = ansint2W
data['vocab'] = vocabulary
pickle_out = open("ProcessedData.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()