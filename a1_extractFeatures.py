import numpy as np
import sys
import argparse
import os
import json
import csv
import string

def calculate_avg_dev(arr):
    ''' This is a helper function to calculate avg and standard deviation
    Parameters:
        arr: array, numbers to be computed
    Returns:
        result: [avg, standard deviation]
    '''
    temp = []
    for num in arr:
        if (num.isdigit() and (not num.isspace())):
            temp.append(int(num))
        else:
            temp.append(0)
    temp = np.array(temp).astype(np.float)
    if (temp.size != 0):
        return [np.mean(temp), np.std(temp)]
    else:
        return [0, 0]

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # print('TODO')
    # TODO: your code here
    first_person = 0
    second_person = 0
    third_person = 0
    coord_conj = 0
    past_verb = 0
    future_verb = 0
    commas = 0
    multi_char_punc = 0
    common_noun = 0
    proper_noun = 0
    adv = 0
    wh_word = 0
    slang_acro = 0
    uppercase = 0
    avg_length = 0
    avg_length_exclude_punc = 0
    num_of_sent = 0
    AOA = []
    IMG = []
    FAM = []
    V_mean_sum = []
    A_mean_sum = []
    D_mean_sum = []

    first_person_file = open("../../Wordlists/First-person", "r")
    first_person_lst = first_person_file.read().lower().splitlines()
    first_person_file.close()

    second_person_file = open("../../Wordlists/Second-person", "r")
    second_person_lst = second_person_file.read().lower().splitlines()
    second_person_file.close()

    third_person_file = open("../../Wordlists/Third-person", "r")
    third_person_lst = third_person_file.read().lower().splitlines()
    third_person_file.close()

    future_tense_words = ["'ll", "will", "gonna"] # need to consider "going to" separately
    common_noun_tags = ["nn", "nns"]
    proper_noun_tags = ["nnp", "nnps"]
    adv_tags = ["rb", "rbr", "rbs"]
    wh_word_tags = ["wdt", "wp", "wp$", "wrb"]

    slang_file = open("../../Wordlists/Slang", "r")
    slang_lst = slang_file.read().lower().splitlines()
    slang_file.close()

    BGL_norms_file = open("../../Wordlists/BristolNorms+GilhoolyLogie.csv", "r")
    BGL_norms = {}
    BGL_norms_reader = csv.reader(BGL_norms_file)
    next(BGL_norms_reader)
    for row in BGL_norms_reader:
        if (row[1] not in BGL_norms):
            BGL_norms[row[1]] = [row[3], row[4], row[5]]
    BGL_norms_file.close()

    warringer_file = open("../../Wordlists/Ratings_Warriner_et_al.csv", "r")
    warringer = {}
    warringer_reader = csv.reader(warringer_file)
    next(warringer_reader)
    for row in warringer_reader:
        if (row[1] not in warringer):
            warringer[row[1]] = [row[2], row[5], row[8]]
    warringer_file.close()

    token_tag = comment.split(" ")

    token_lst = []
    tag_lst = []
    token_len = 0
    token_exclude_punc = 0
    num_char = 0
    for t in token_tag:
        if (t.isspace()):
            continue
        if (t == "\n"):
            token_lst.append("\n")
            tag_lst.append("\n")
            num_of_sent += 1
        else:
            idx = t.find("/")
            token = t[:idx]
            tag = t[idx+1:]

            if (token.isspace() and tag.isspace()):
                continue

            # 14. Number of words in uppercase (≥ 3 letters long)
            if (len(token_lst) >= 3):
                if (token.isupper()):
                    uppercase += 1

            # lower token and tag, since my tags are all lowercased.
            token = token.lower()
            tag = tag.lower()

            if not (all(i in string.punctuation for i in token)):
                num_char += len(token)
                token_exclude_punc += 1

            # 1. Number of first-person pronouns
            if (token in first_person_lst):
                first_person += 1

            # 2. Number of second-person pronouns
            if (token in second_person_lst):
                second_person += 1

            # 3. Number of third-person pronouns
            if (token in third_person_lst):
                third_person += 1

            # 4. Number of coordinating conjunctions
            if (tag == 'cc'):
                coord_conj += 1

            # 5. Number of past-tense verbs
            if (tag == 'vbd'):
                past_verb += 1
     
            # 6. Number of future-tense verbs
            if (tag == 'vb'):
                if (len(token_lst) > 1) and (token_lst[-1] in future_tense_words):
                    future_verb += 1
                # going+to+vb
                if (len(token_lst) > 2) and (token_lst[-2] == "going" and token_lst[-1] == to):
                    future_verb += 1

            # 7. Number of commas
            if (token == ","):
                commas += 1

            # 8. Number of multi-character punctuation tokens
            if (len(token) >= 2):
                if (all(i in string.punctuation for i in token)):
                    multi_char_punc += 1
       
            # 9. Number of common nouns
            if (tag in common_noun_tags):
                common_noun += 1
      
            # 10. Number of proper nouns
            if (tag in proper_noun_tags):
                proper_noun += 1
      
            # 11. Number of adverbs
            if (tag in adv_tags):
                adv += 1

            # 12. Number of wh- words
            if (tag in wh_word_tags):
                wh_word += 1
       
            # 13. Number of slang acronyms
            if (token in slang_lst):
                slang_acro += 1
     
            if (token in BGL_norms):
                AOA.append(BGL_norms[token][0])
                IMG.append(BGL_norms[token][1])
                FAM.append(BGL_norms[token][2])

            if (token in warringer):
                V_mean_sum.append(warringer[token][0])
                A_mean_sum.append(warringer[token][1])
                D_mean_sum.append(warringer[token][2])

            token_lst.append(token)
            tag_lst.append(tag)
            token_len += 1

    # 17. Number of sentences.
    # num_of_sent = 0
    if (token_lst and token_lst[-1] != "\n"):
        token_lst.append("\n")
        tag_lst.append("\n")
        num_of_sent += 1

    # 15. Average length of sentences, in tokens
    if (num_of_sent > 0):
        avg_length = token_len / num_of_sent

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    if (token_exclude_punc > 0):
        avg_length_exclude_punc = num_char / token_exclude_punc

    feats = np.zeros(29)
    feats[0] = first_person
    feats[1] = second_person
    feats[2] = third_person
    feats[3] = coord_conj
    feats[4] = past_verb
    feats[5] = future_verb
    feats[6] = commas
    feats[7] = multi_char_punc
    feats[8] = common_noun
    feats[9] = proper_noun
    feats[10] = adv
    feats[11] = wh_word
    feats[12] = slang_acro
    feats[13] = uppercase
    feats[14] = avg_length
    feats[15] = avg_length_exclude_punc
    feats[16] = num_of_sent

    AOA_result = calculate_avg_dev(AOA)
    IMG_result = calculate_avg_dev(IMG)
    FAM_result = calculate_avg_dev(FAM)
    V_result = calculate_avg_dev(V_mean_sum)
    A_result = calculate_avg_dev(A_mean_sum)
    D_result = calculate_avg_dev(D_mean_sum)
    feats[17] = AOA_result[0]
    feats[18] = IMG_result[0]
    feats[19] = FAM_result[0]
    feats[20] = AOA_result[1]
    feats[21] = IMG_result[1]
    feats[22] = FAM_result[1]
    feats[23] = V_result[0]
    feats[24] = A_result[0]
    feats[25] = D_result[0]
    feats[26] = V_result[1]
    feats[27] = A_result[1]
    feats[28] = D_result[1]

    return feats


def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here
    alt_feats = np.load("../feats/Alt_feats.dat.npy")
    alt_ids = open("../feats/Alt_IDs.txt").read().strip().split("\n")
    alt_idx = {}
    for i in range(len(alt_ids)):
        alt_idx[alt_ids[i].strip()] = i

    center_feats = np.load("../feats/Center_feats.dat.npy")
    center_ids = open("../feats/Center_IDs.txt").read().strip().split("\n")
    center_idx = {}
    for i in range(len(center_ids)):
        center_idx[center_ids[i].strip()] = i 

    left_feats = np.load("../feats/Left_feats.dat.npy")
    left_ids = open("../feats/Left_IDs.txt").read().strip().split("\n")
    left_idx = {}
    for i in range(len(left_ids)):
        left_idx[left_ids[i].strip()] = i

    right_feats = np.load("../feats/Right_feats.dat.npy")
    right_ids = open("../feats/Right_IDs.txt").read().strip().split("\n")
    right_idx = {}
    for i in range(len(right_ids)):
        right_idx[right_ids[i].strip()] = i

    for i in range(len(data)):
        line = data[i]
        comment = line['body']
        id = line['id'].lower()
        cat = line['cat'].lower()
       
        feats[i, 0:29] = extract1(comment)

        if cat == "alt":
            feats[i, 29:173] = alt_feats[alt_idx[id]]
            feats[i, 173] = 3
        elif cat == "center":
            feats[i, 29:173] = center_feats[center_idx[id]]
            feats[i, 173] = 1
        elif cat == "left":
            feats[i, 29:173] = left_feats[left_idx[id]]
            feats[i, 173] = 0
        elif cat == "right":
            feats[i, 29:173] = right_feats[right_idx[id]]
            feats[i, 173] = 2

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

