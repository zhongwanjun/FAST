import json
import pickle as pkl
from tqdm import tqdm
import nltk
import time
import re
import os
import random
from nltk.corpus import stopwords
import math
import shutil
import pathos
from fuzzywuzzy import fuzz

def extract_documents_sentence(fileNames):
    def clean_sentence(sentences):
        result = []
        stopWords = set(stopwords.words('english'))
        for sentence in tqdm(sentences, desc="cleaning sentence"):
            sentence = sentence.strip().lower()
            sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
            sentence = re.sub(r"\'s", " \'s", sentence)
            sentence = re.sub(r"\'ve", " \'ve", sentence)
            sentence = re.sub(r"n\'t", " n\'t", sentence)
            sentence = re.sub(r"\'re", " \'re", sentence)
            sentence = re.sub(r"\'d", " \'d", sentence)
            sentence = re.sub(r"\'ll", " \'ll", sentence)
            sentence = re.sub(r",", " , ", sentence)
            sentence = re.sub(r"!", " ! ", sentence)
            sentence = re.sub(r"\(", " ( ", sentence)
            sentence = re.sub(r"\)", " ) ", sentence)
            sentence = re.sub(r"\?", " ? ", sentence)
            sentence = re.sub(r"\s{2,}", " ", sentence)
            sentence = re.sub("[0123456789]", "", sentence)
            sentence = sentence.split()
            sentence = " ".join([w for w in sentence if w not in stopWords])
            result.append(sentence)
        return result

    documents = []
    for pathDir in fileNames:
        result = []
        data = [json.loads(s) for s in open("{}".format(pathDir), 'r').readlines()]
        for line in data:
            # if line["split"] == "train":
            if line["split"] == "val" and line['label']=='human':
                result.append(line["article"])
        documents.extend(result)
    print("num of documents is {}".format(len(documents)))
    sentences, pairs, dict, index = [], [], {}, 0
    for document in tqdm(documents, desc="extracting sentences"):
        document = re.sub("\n+", ". ", document)
        sentence = nltk.tokenize.sent_tokenize(document)
        sentences.extend(sentence)
        for i in range(len(sentence)-1):
            dict[index] = index + 1
            pairs.append([index, index+1])
            index += 1
        index += 1
    print("num of sentences is {}".format(len(sentences)))
    sentencesClean = clean_sentence(sentences)
    return sentences,sentencesClean, pairs, dict


def sampling(sentencesClean, sentences, pairs, dict, NSN=10000, processNum=1):
    def remove_sentence_which_is_the_same_with_next(sentences, dict, target, negativeSamplingIds):
        result = []
        for id in negativeSamplingIds:
            if len(sentences[id].split()) == 0:
                continue
            if sentences[id] != sentences[dict[target]]:
                result.append(id)
        return result

    def sampling_with_fuzzywuzzy(pair,sentencesClean, sentences,negativeSamplingIds):
        maxValue, result = -1, []
        sentence = sentencesClean[pair[1]]
        for id in negativeSamplingIds:
            # negativeWordSet = set(sentencesClean[id].split())
            neg_sentence = sentencesClean[id]
            score = fuzz.ratio(sentence,neg_sentence)
            if score > maxValue:
                maxValue = score
                result = [id]
            elif score == maxValue:
                result.append(id)
        if len(result)>1:
            result = sorted(result,key=lambda k:abs(len(sentences[k].split())-len(sentence.split())),reverse=True)
        # pair.append(random.choice(result))
        pair.append(result[0])
        return pair

    def sampling_with_most_word_cooccurrence(pair, sentencesClean, negativeSamplingIds):
        maxValue, result = -1, []
        words = sentencesClean[pair[1]].split()
        for id in negativeSamplingIds:
            match = 0
            negativeWordSet = set(sentencesClean[id].split())
            for word in words:
                if word in negativeWordSet:
                    match += 1
            score = 1.0*match/math.pow(len(negativeWordSet), 0.75)
            if score > maxValue:
                maxValue = score
                result = [id]
            elif score == maxValue:
                result.append(id)

        pair.append(random.choice(result))
        return pair

    def sub_sampling(pairs, dict, sentencesClean, begin, end):
        result = []
        for i in tqdm(range(begin, end), desc="sampling....."):
            pair = pairs[i]
            if len(sentencesClean[pair[0]].split()) == 0 or len(sentencesClean[pair[1]].split()) == 0:
                continue
            negativeSamplingIds = random.choices(range(len(sentencesClean)), k=NSN)
            negativeSamplingIds = remove_sentence_which_is_the_same_with_next(sentencesClean, dict, pair[0], negativeSamplingIds)
            result.append(sampling_with_fuzzywuzzy(pair, sentencesClean,sentences, negativeSamplingIds))
        return result

    pool = pathos.multiprocessing.Pool(processes=processNum)
    results, resultsObjects, eachProcessNum = [], [], len(pairs) // processNum
    for i in range(processNum - 1):
        resultsObjects.append(pool.apply_async(func=sub_sampling, args=[pairs, dict, sentencesClean, i * eachProcessNum, (i + 1) * eachProcessNum]))
    resultsObjects.append(pool.apply_async(func=sub_sampling, args=[pairs, dict, sentencesClean, (processNum - 1) * eachProcessNum, len(pairs)]))
    pool.close()
    pool.join()

    for resultsObject in resultsObjects:
        results.extend(resultsObject.get())

    return results

def basic_clean_sentence(sentence):
    sen = sentence.strip().replace('\t',' ').replace('\n','. ')
    sen = re.sub(r'[^a-zA-Z0-9,.\'\`!?]+', ' ', sen)
    return sen

def generate_and_delete(sentences, pairDone):
    results = pairDone
    sentence = sentences
    index = 0
    with open("/mnt/wanjun/data/realnews_human_val.tsv", 'w', encoding='utf-8') as outf:
        for line in results:
            outf.write("{}\t{}\t{}\t{}\n".format(index, 1, basic_clean_sentence(sentence[line[0]]), basic_clean_sentence(sentence[line[1]])))
            index += 1
            outf.write("{}\t{}\t{}\t{}\n".format(index, 0, basic_clean_sentence(sentence[line[0]]), basic_clean_sentence(sentence[line[2]])))
            index += 1
        outf.close()


if __name__ == '__main__':
    sentences,sentencesClean, pairs, dict = extract_documents_sentence(["/home/v-wanzho/wanjun/deepfake/data/realnews/p=0.96.jsonl"])
    pairDone = sampling(sentencesClean,sentences, pairs, dict, NSN=10000, processNum=10)
    generate_and_delete(sentences, pairDone)
