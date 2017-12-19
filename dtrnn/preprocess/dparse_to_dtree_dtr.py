#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 23:39:50 2017

"""

from dtree_util_dtr import *
import sys, cPickle, random, os, subprocess

random.seed(2017)

SENT_FILE = '../data/dev_dump.txt'
RAW_PARSES_FILE = '../data/raw_parses_for_tain_sent5'
STANFORD_LEXPARSER = '../../stanford_parser/lexparser.sh '
FINAL_SPLIT_FILE = '../data/final_split_train5'


# - given a text file where each line is a sentence, use the
#   stanford dependency parser to create a dependency parse tree for each sentence
def dparse(sentence_file):
    out_file = open(RAW_PARSES_FILE, 'w')
    
    # change these paths to point to your stanford parser.
    # make sure to use the lexparser.sh file in this directory instead of the default!
    parser_out = os.popen(STANFORD_LEXPARSER + sentence_file).readlines()
    
    for line in parser_out:
        out_file.write(line)

    out_file.close()


# - function that parses the resulting stanford parses
#   e.g., "nsubj(finalized-5, john-1)"
def split_relation(text):
    rel_split = text.split('(')
    rel = rel_split[0]
    deps = rel_split[1][:-1]
    if len(rel_split) != 2:
        print 'error ', rel_split
        sys.exit(0)

    else:
        dep_split = deps.split(',')

        # more than one comma (e.g. 75,000-19)
        if len(dep_split) > 2:

            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

            print 'fixed: ', fixed
            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )

        return rel, final_deps


# - given a list of all the split relations in a particular sentence,
#   create a dtree object from that list
def make_tree(plist):

    # identify number of tokens
    max_ind = -1
    for rel, deps in plist:
        for ind, word in deps:
            if ind > max_ind:
                max_ind = ind

    # load words into nodes, then make a dependency tree
    nodes = [None for i in range(0, max_ind + 1)]
    for rel, deps in plist:
        for ind, word in deps:
            nodes[ind] = word

    tree = dtree(nodes)

    # add dependency edges between nodes
    for rel, deps in plist:
        par_ind, par_word = deps[0]
        kid_ind, kid_word = deps[1]
        tree.add_edge(par_ind, kid_ind, rel)

    return tree  



# - given all dependency parses of a dataset as well as that dataset (in the same order),
#   dumps a processed dataset that can be fed into DT-RNN:
#   (vocab, list of dep. relations, list of texts, and dict of {fold: list of dtrees})
def process_text_file(raw_parses, sentence_file):

    parses = open(raw_parses, 'r')
    split = makeSplits('', sentence_file)
    parse_text = []
    new = False
    cur_parse = []
    for line in parses:

        line = line.strip()

        if not line:
            new = True

        if new:
            parse_text.append(cur_parse)
            cur_parse = []
            new = False

        else:
            # print line
            rel, final_deps = split_relation(line)
            cur_parse.append( (rel, final_deps) )

    print len(parse_text)

    # make mapping 
    count = 0
    tree_dict = {}
    for key in split:
        hist = split[key]
        tree_dict[key] = []
        for text, qid in hist:
            for i in range(0, len(text)):

                tree = make_tree(parse_text[count])
                tree.dist = i
                tree.qid = qid
                tree_dict[key].append(tree)
                count += 1

    vocab = []
    rel_list = []


    for key in tree_dict:
        print 'processing ', key
        qlist = tree_dict[key]
        for tree in qlist:
            for node in tree.get_nodes():
                if node.word not in vocab:
                    vocab.append(node.word)
                
                node.ind = vocab.index(node.word)

                for ind, rel in node.kids:
                    if rel not in rel_list:
                        rel_list.append(rel)

    print 'rels: ', len(rel_list)
    print 'vocab: ', len(vocab)

    cPickle.dump((vocab, rel_list, tree_dict), open(FINAL_SPLIT_FILE, 'wb'))
    
    
def makeSplits(qa_folder, sentence_file):
    splits = {}
    splits['train'] = makeSplit(sentence_file)
    return splits

def makeSplit(sentence_file):

    sents = open(sentence_file, 'r')
    qid = 1
    split = []
    for sent in sents:
        split.append([[sent], qid])
        qid += 1
        
    return split


if __name__ == '__main__':
    sentence_file = SENT_FILE
    dparse(sentence_file)
    process_text_file(RAW_PARSES_FILE, sentence_file)
