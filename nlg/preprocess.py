# Copyright (c) Facebook, Inc. and its affiliates.

import json
import os
import copy
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/adl-final-dst-with-chit-chat-seen-domains/data-0614/data-0614", type=str, required=False, help="path to data")
    args = parser.parse_args()

    datafolder = args.data
    inmc = {}
    for folder in ["train", "dev"]:
        inlm = []
        inmc['data'] = []
        fns = os.listdir(os.path.join(datafolder, folder))
        fns.sort()
        for fn in fns:
            with open(os.path.join(datafolder, folder, fn), "r", encoding='utf8') as f:
                data = json.load(f)

            for dialogue in data:
                context = ''
                context4mc = ''

                for turn in dialogue['turns']:
                    speaker = turn['speaker']
                    utterance = turn['utterance']

                    if speaker == 'SYSTEM':
                        target = ''
                        target += '<|context|> ' + context + ' <|endofcontext|> '
                        target += '<|response|> ' + utterance + ' <|endofresponse|> '
                        chitchats_good = []
                        chitchats_bad = []
                        if 'beginning' in turn:
                            for chitchat in turn['beginning']:
                                if chitchat['label'] == 'good':
                                    chitchats_good += [chitchat['candidate']]
                                else:
                                    chitchats_bad += [chitchat['candidate']]
                                inmc['data'] += [{
                                    'context': context4mc,
                                    'response': utterance,
                                    'chitchat': chitchat['candidate'],
                                    'answer': 0 if chitchat['label'] == 'good' else 1
                                }]
                        if 'end' in turn:
                            for chitchat in turn['end']:
                                if chitchat['label'] == 'good':
                                    chitchats_good += [chitchat['candidate']]
                                else:
                                    chitchats_bad += [chitchat['candidate']]
                                inmc['data'] += [{
                                    'context': context4mc,
                                    'response': utterance,
                                    'chitchat': chitchat['candidate'],
                                    'answer': 2 if chitchat['label'] == 'good' else 1
                                }]

                        inlm += ['<|endoftext|> ' + target  + '<|chitchat|> ' + chitchat + ' <|endofchitchat|> ' + ' <|endoftext|>' for chitchat in chitchats_good]
                        if len(chitchats_good) == 0:
                            inlm += ['<|endoftext|> ' + target + '<|endoftext|>']
                    
                    context += '<|' + speaker.lower() + '|> ' + utterance
                    context4mc += speaker.lower() + ': '+ utterance + ' '

        random.shuffle(inlm)
        with open("lm.input1."+folder+".txt", "w", encoding='utf8') as f: #SimpleTOD
            f.write('\n'.join(inlm))

        with open("arranger_input." + folder + ".json", "w", encoding='utf8') as f:
            json.dump(inmc, f, indent=1)

if __name__ == '__main__':
    random.seed(42)
    main()
