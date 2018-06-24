import argparse

def main(args):
    with open(args.input, 'r', encoding='utf-8') as f_in:
        with open(args.output, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                # Read / parse line
                lang, sent = line.strip().split('\t')

                # Write features
                f_out.write(lang)
                # Write tab character
                f_out.write('\t')
                # Treat each word as a feature.
                words = sent.split(' ')
                for word in words:
                    f_out.write(word)
                    f_out.write(' ')
                # Write final endline
                f_out.write('\n')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input file')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    args = parser.parse_args()

    main(args)
    
