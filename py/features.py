from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
 
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import *

import argparse

def main(args):
    with open(args.input, 'r', encoding='utf-8') as f_in:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # Read / parse line
                lang, sent = line.strip().split('\t')

                # Write features
                f_out.write(lang)
                f_out.write('\t')
                if lang == 'cmn' or lang == 'jpn':
                    # Treat each character as a feature, since there are no words.
                    for ch in sent:
                        f_out.write(ch)
                        f_out.write(' ')
                else:
                    # Treat each word as a feature.
                    words = sent.split(' ')
                    for word in words:
                        f_out.write(word)
                        f_out.write(' ')

                # add the features for the character ngrams.
                if args.char_ngram_upto > 0:
                    n = len(sent)
                    for k in xrange(args.char_ngram_upto):
                        for i in xrange(n - k):
                            x = sent[i:i + k + 1].replace(' ', '_')
                            f_out.write(x)
                            f_out.write(' ')
                     
                # write the final character.                
                f_out.write('\n')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input file')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    parser.add_argument('--char_ngram_upto', type=int, default=0)
    args = parser.parse_args()

    main(args)
    
