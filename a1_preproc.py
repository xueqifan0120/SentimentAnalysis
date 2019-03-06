import sys
import argparse
import os
import json
import re
import spacy
import html
import spacy

indir = '/u/cs401/A1/data/';
nlp = spacy.load('en', disable=['parser', 'ner'])

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    if 1 in steps:
        #print('TODO')
        modComm = comment.replace("\n", " ").replace("\r", " ")
    if 2 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 2):
            cmt = comment
        else:
            cmt = modComm
        modComm = html.unescape(cmt)
    if 3 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 3):
            cmt = comment
        else:
            cmt = modComm
        pattern = re.compile(r"(?:http\S+|www\S*)")
        modComm = pattern.sub('',cmt)
    if 4 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 4):
            cmt = comment
        else:
            cmt = modComm
        import string
        punc = string.punctuation
        # do not split on '
        punc = punc.replace("'", "")
        # do not split on multi punc
        comms = [item.strip() for item in 
                re.split("("+"[\s{}]+".format(re.escape(punc))+")", cmt) 
                if len(item) > 0]
        # check abbr, abbr always ends at .
        file = open('/u/cs401/Wordlists/abbrev.english', 'r')
        abbrs = file.read().strip().split("\n")
        modComms = []
        i = 0
        while i < len(comms):
            if (i == len(comms) - 1):
                modComms.append(comms[i])
                break
            if (len(comms[i + 1]) != 0):
                curr = comms[i] + comms[i + 1][0]
            else:
                curr = comms[i]
            if (curr in abbrs):
                modComms.append(curr)
                if (len(comms[i+1]) > 1):
                    comms[i+1] = comms[i+1][1:]
                else:
                    i += 1
            else:
                modComms.append(comms[i])
            i += 1
        file.close()
        # check pn_abbr, this contains special case, e.g.,i.e.
        file = open('/u/cs401/Wordlists/pn_abbrev.english', 'r')
        abbrs = file.read().strip().split("\n")
        newComms = []
        for i in range(len(modComms)):
            if (i == 0):
                newComms.append(modComms[i])
                continue
            added = False
            if (modComms[i] == '.'):
                curr = modComms[i-1] + modComms[i]
                if (curr in abbrs):
                    newComms = newComms[:-1]
                    newComms.append(curr)
                    added = True
            if (i >= 3 and modComms[i] == '.' and modComms[i-2] == '.'):
                curr = modComms[i-3] + modComms[i-2] + modComms[i-1] + modComms[i]
                if (curr in abbrs):
                    newComms = newComms[:-3]
                    newComms.append(curr)
                    added = True
            if (not added):
                newComms.append(modComms[i])
        modComm = ' '.join(newComms)
    if 5 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 5):
            cmt = comment
        else:
            cmt = modComm
        file = open('/u/cs401/Wordlists/clitics', 'r')
        clitics = file.read().strip().replace("\n", "|").strip()
        comms = [item.strip() for item in 
                re.split("(\\b(?:"+clitics+")\\b)", cmt) 
                if len(item) > 0]
        file.close()
        # Deal with plurals (dogs')
        comms = ' '.join(comms).split(' ')
        modComms = []
        for comm in comms:
            if (len(comm) > 1 and comm[-1] == "'"):
                modComms.append(comm[:-1])
                modComms.append(comm[-1])
            else:
                modComms.append(comm)
        modComm = ' '.join(modComms)
    if 6 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 6):
            cmt = comment
        else:
            cmt = modComm
        utt = nlp(cmt)
        modComm = ''
        for token in utt:
            modComm += token.text + "/" + token.tag_ + " "
    if 7 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 7):
            cmt = comment
        else:
            cmt = modComm
        file = open('/u/cs401/Wordlists/StopWords', 'r')
        stopwords = file.read().strip().split("\n")
        comments = cmt.split(" ")
        modComms = []
        for cmt in comments:
            if (cmt == ''):
                continue
            idx = cmt.rfind("/")
            if (idx == -1):
                token = cmt
            else:
                token = cmt[:idx]
            if (token not in stopwords):
                modComms.append(cmt)
        file.close()
        modComm = ' '.join(modComms)
    if 8 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 8):
            cmt = comment
        else:
            cmt = modComm
        modComms = []
        comments = cmt.split(" ")
        for cmt in comments:
            if (cmt == ''):
                continue
            idx = cmt.rfind("/")
            if (idx == -1):
                token = cmt
                tag = ''
            else:
                token = cmt[:idx]
                tag = cmt[idx+1:]
            utt = nlp(token)
            for t in utt:
                # if lema begins with '-' and token doesn't, keep token
                if (t.lemma_ and t.lemma_[0] == "-" and token and token[0] != "-"):
                    modComms.append(token + "/" + tag)
                else:
                    modComms.append(t.lemma_ + "/" + tag)
        modComm = ' '.join(modComms)
    if 9 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 9):
            cmt = comment
        else:
            cmt = modComm
        file = open('/u/cs401/Wordlists/abbrev.english', 'r')
        abbrs = file.read().strip().lower().split("\n")
        file.close()
        comms = cmt.split(" ")
        ends = [".", "!", "?"]
        modComms = []
        for i in range(len(comms)):
            if (i == 0):
                modComms.append(comms[i])
                continue
            curr = comms[i]
            idx = curr.rfind("/")
            if (idx == -1):
                curr_token = curr
            else:
                curr_token = curr[:idx]
            prev = comms[i-1]
            idx = prev.rfind("/")
            if (idx == -1):
                prev_token = prev
            else:
                prev_token = prev[:idx]
            checked = prev_token + curr_token
            if (curr_token not in ends):
                modComms.append(curr)
            elif (checked.lower() in abbrs):
                modComms.append(curr)
            else:
                modComms.append(curr)
                modComms.append("\n")
        file = open('/u/cs401/Wordlists/pn_abbrev.english', 'r')
        pn_abbrs = file.read().strip().lower().split("\n")
        file.close()
        # check for pn_abbrev.English
        newComms = []
        for i in range(len(modComms)):
            if (i < 1):
                newComms.append(modComms[i])
                continue
            added = False
            curr = modComms[i-1] + modComms[i]
            if (curr in pn_abbrs):
                newComms = newComms[:-1]
                newComms.append(curr)
                added = True
            if (i >= 3):
                curr = modComms[i-3] +  modComms[i-2] + modComms[i-1] + modComms[i]
                if (curr in pn_abbrs):
                    newComms = newComms[:-3]
                    newComms.append(curr)
                    added = True
            if (not added):
                newComms.append(modComms[i])

        modComm = ' '.join(newComms)
    if 10 in steps:
        # print('TODO')
        cmt = ''
        if (min(steps) == 10):
            cmt = comment
        else:
            cmt = modComm
        # do not lower case PoS
        comms = modComm.split(" ")
        newComms = []
        for comm in comms:
            idx = comm.rfind("/")
            if (idx == -1):
                newComms.append(comm.lower())
            else:
                token = comm[:idx].lower()
                tag = comm[idx+1:]
                newComms.append(token+"/"+tag)
        modComm = ' '.join(newComms)
        
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            max_line = int(args.max)
            curr = 0 #current sampled line number
            looped = False
            while curr < max_line:
                if (looped == True):
                    idx = 0
                else:
                    idx = args.ID[0] % len(data)
                for i in range(idx, min(idx + max_line, len(data))):
                    if (curr == max_line):
                        break
                    line = data[i]
                    # TODO: read those lines with something like `j = json.loads(line)`
                    j = json.loads(line)
                    # TODO: choose to retain fields from those lines that are relevant to you
                    j = {key: j[key] for key in ('body', 'id')}
                    # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                    j['cat'] = fullFile.split('/')[-1]
                    # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                    preproc = preproc1(j['body'])
                    # TODO: replace the 'body' field with the processed text
                    j['body'] = preproc
                    # TODO: append the result to 'allOutput'
                    allOutput.append(j)
                    curr += 1
                looped = True
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
