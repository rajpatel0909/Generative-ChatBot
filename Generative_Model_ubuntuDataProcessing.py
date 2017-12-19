import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

contexts = []
responses = []
cntr = 0
filename = 'C:/MyStuff/SEM3/DL/Project1/datasets/ubuntu-ranking-dataset-creator-master/udc.tar/data/train.csv'
file = open(filename, encoding="utf8")
csvReader = csv.reader(file)
for row in csvReader:
    contexts.append(row[0])
    responses.append(row[1])
    cntr += 1
    
# lengths = []
# for i in contexts:
#     lengths.append(len(i))
#     
# # fixed bin size
# bins = np.arange(-100, 100, 5) # fixed bin size
# plt.xlim([min(lengths)-5, max(lengths)+5])
# plt.hist(lengths, bins=bins, alpha=0.5)
# plt.title('Random Gaussian data (fixed bin size)')
# plt.xlabel('variable X (bin size = 5)')
# plt.ylabel('count')
# plt.axis([0, 150, 0, 9000])
# plt.show()

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
    print(responses[i])

for que in contexts:
    replacedQues.append(replaceText(que))
     
for ans in responses:
    replacedAns.append(replaceText(ans))

print('replaced6')
for i in range(2):
    print(replacedQues[i])
    print(replacedAns[i])

# removing conversations with large difference in lengths
lenDiff = 20
consiseQues = []
consiseAns = []

for i in range(0, len(replacedQues)):
    if abs(len(replacedAns[i]) - len(replacedQues[i])) <= lenDiff:
        consiseQues.append(replacedAns[i])
        consiseAns.append(replacedQues[i])       

print(len(replacedQues))
print(len(replacedAns))
print(len(consiseAns))
print(len(consiseQues))

vocabulary = {}
for que in contexts:
    for w in que.split():
        if w not in vocabulary:
            vocabulary[w] = 1
        else:
            vocabulary[w] += 1
             
for ans in responses:
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

unkcntr = 0

queInt = []
for que in consiseQues:
    ints = []
    for w in que.split():
        if w not in queW2int:
            unkcntr += 1
            ints.append(queW2int['<UNK>'])
        else:
            if w in vocabulary:
                if vocabulary[w] > 15 and vocabulary[w] < 100000:
                    ints.append(queW2int[w])
            else:
                ints.append(queW2int[w])
    queInt.append(ints)
    
ansInt = []
for ans in consiseAns:
    ints = []
    for w in ans.split():
        if w not in ansW2int:
            unkcntr += 1
            ints.append(ansW2int['<UNK>'])
        else:
            if w in vocabulary:
                if vocabulary[w] > 15 and vocabulary[w] < 100000:
                    ints.append(ansW2int[w])
            else:
                ints.append(ansW2int[w])
    ansInt.append(ints)
    
    
train_valid_split = int(len(queInt)*0.15)

train_questions = queInt[train_valid_split:]
train_answers = ansInt[train_valid_split:]

valid_questions = queInt[:train_valid_split]
valid_answers = ansInt[:train_valid_split]

PAD = queW2int['<PAD>']
for i in range(0, len(train_questions)):
    ans_len = len(train_answers[i])
    que_len = len(train_questions[i])
    extra_array = [PAD] * abs(ans_len - que_len)
    if ans_len > que_len:
        train_questions[i] += extra_array
    else:
        train_answers[i] += extra_array
        
    if i < 10:
        print (train_answers[i])
        print()
        print (train_questions[i])
        print()

for i in range(0, len(valid_questions)):
    ans_len = len(valid_answers[i])
    que_len = len(valid_questions[i])
    extra_array = [PAD] * abs(ans_len - que_len)
    if ans_len > que_len:
        valid_questions[i] += extra_array
    else:
        valid_answers[i] += extra_array
        

print (len(train_questions))

for i in range(25):
    print(i, queint2W[i], "->", queW2int[queint2W[i]])
    
for i in range(25):
    print(i, ansint2W[i], "->", ansW2int[ansint2W[i]])

print('unkcntr ', unkcntr)
#store train_questions train_answers valid_questions and valid_answers and word2int and int2word
data = {}
data['train_questions'] = train_questions
data['train_answers'] = train_answers
data['valid_questions'] = valid_questions
data['valid_answers'] = valid_answers
data['queW2int'] = queW2int
data['queint2W'] = queint2W
data['ansW2int'] = ansW2int
data['ansint2W'] = ansint2W
data['vocab'] = vocabulary
pickle_out = open("vubuntuProcessedData.pickle","wb")
pickle.dump(data, pickle_out)
pickle_out.close()