# Huffman encoder for birds/flowers dataset
# 
# Compression application using static Huffman coding
# 
# Usage: python huffman-compress.py InputFile OutputFile
# Then use the corresponding huffman-decompress.py application to recreate the original input file.
# Note that the application uses an alphabet of 257 symbols - 256 symbols for the byte values
# and 1 symbol for the EOF marker. The compressed file format starts with a list of 257
# code lengths, treated as a canonical code, and then followed by the Huffman-coded data.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-huffman-coding
# https://github.com/nayuki/Reference-Huffman-coding
# 
# fork from: https://github.com/nayuki/Reference-Huffman-coding/blob/master/python/huffman-compress.py

import contextlib, sys, argparse, json
import HuffmanCoding as huffmancoding
python3 = sys.version_info.major >= 3

## ---------- write the data ---------- ##
# Command line main application function.
def main_write(args):
    # Handle command line arguments
    train_json, test_json = args.train_json, args.test_json
    output_file = args.output_file
    keys = args.keys.split(':')
    
    # Read input file once to compute symbol frequencies.
    # The resulting generated code is optimal for static Huffman coding and also canonical.
    freqs = get_frequencies_json(train_json, keys)
    freqs.increment(256)  # EOF symbol gets a frequency of 1
    code = freqs.build_code_tree()
    canoncode = huffmancoding.CanonicalCode(tree=code, symbollimit=freqs.get_symbol_limit())
    # Replace code tree with canonical one. For each symbol,
    # the code value may change but the code length stays the same.
    code = canoncode.to_code_tree()
    
    # Read input file again, compress with Huffman coding, and write output file
    with open(test_json, "r") as inp, \
            contextlib.closing(huffmancoding.BitOutputStream(open(output_file, "wb"))) as bitout:
        data = json.load(inp)
        write_code_len_table(bitout, canoncode)
        compresser = Compression(bitout, code)
        traverse_data(data, compresser.step, keys)
        compresser.write_eof()
        print({'bit':compresser.bit_total, 'char':compresser.char_total})

# Returns a frequency table based on the bytes in the given file.
# Also contains an extra entry for symbol 256, whose frequency is set to 0.

def traverse_data(_d, func, keys):
    if isinstance(_d, list):
        for _d_temp in _d:
            traverse_data(_d_temp, func, keys)
    elif isinstance(_d, dict):
        _d_temp = _d[keys[0]]
        keys = keys[1:]
        traverse_data(_d_temp, func, keys)
    elif isinstance(_d, str):
        for _c in _d:
            func(ord(_c))
    else:
        raise ValueError


def get_frequencies_json(filepath:str, keys:str):
    freqs = huffmancoding.FrequencyTable([0] * 257)
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    traverse_data(data, freqs.increment, keys)
    return freqs

def write_code_len_table(bitout, canoncode):
    for i in range(canoncode.get_symbol_limit()):
        val = canoncode.get_code_length(i)
        # For this file format, we only support codes up to 255 bits long
        if val >= 256:
            raise ValueError("The code for a symbol is too long")
        
        # Write value as 8 bits in big endian
        for j in reversed(range(8)):
            bitout.write((val >> j) & 1)

class Compression(object):
    def __init__(self, bitout, code):
        self.enc = huffmancoding.HuffmanEncoder(bitout)
        self.enc.codetree = code
        self.bit_total = 0
        self.char_total = 0

    def step(self, c):
        self.enc.write(c)
        self.bit_total += len(self.enc.codetree.codes[c])
        self.char_total += 1

    def write_eof(self):
        self.enc.write(256)

def compress(code, inp, bitout):
    enc = huffmancoding.HuffmanEncoder(bitout)
    enc.codetree = code
    while True:
        b = inp.read(1)
        if len(b) == 0:
            break
        enc.write(b[0] if python3 else ord(b))
    enc.write(256)  # EOF


## -------- decompress the bit data ------ ##

def main_decompress(args):
	# Handle command line arguments
	inputfile, outputfile = args.input_file, args.output_file
	
	# Perform file decompression
	with open(inputfile, "rb") as inp, open(outputfile, "wb") as out:
		bitin = huffmancoding.BitInputStream(inp)
		canoncode = read_code_len_table(bitin)
		code = canoncode.to_code_tree()
		decompress(code, bitin, out)


def read_code_len_table(bitin):
	def read_int(n):
		result = 0
		for _ in range(n):
			result = (result << 1) | bitin.read_no_eof()  # Big endian
		return result
	
	codelengths = [read_int(8) for _ in range(257)]
	return huffmancoding.CanonicalCode(codelengths=codelengths)


def decompress(code, bitin, out):
	dec = huffmancoding.HuffmanDecoder(bitin)
	dec.codetree = code
	while True:
		symbol = dec.read()
		if symbol == 256:  # EOF symbol
			break
		out.write(bytes((symbol,)) if python3 else chr(symbol))


## ------- statistic the bit rate ---------- ##

def main_statistic(args):
    # Handle command line arguments
    train_file, test_file = args.train_json, args.test_json
    keys = args.keys.split(":")
    # Read input file once to compute symbol frequencies.
    # The resulting generated code is optimal for static Huffman coding and also canonical.
    freqs = get_frequencies_json(train_file, keys)
    freqs.increment(256)  # EOF symbol gets a frequency of 1
    code = freqs.build_code_tree()
    canoncode = huffmancoding.CanonicalCode(tree=code, symbollimit=freqs.get_symbol_limit())
    # Replace code tree with canonical one. For each symbol,
    # the code value may change but the code length stays the same.
    code = canoncode.to_code_tree()
    
    # Read input file again, compress with Huffman coding, and write output file
    bit_total = statistic_json_bit_rate(test_file, keys, code.codes)
    print(bit_total)

def main_statistic_end_to_end(args):
    # Handle command line arguments
    train_file, test_file = args.train_json, args.test_json
    keys = args.keys.split(":")
    # Read input file once to compute symbol frequencies.
    # The resulting generated code is optimal for static Huffman coding and also canonical.
    freqs = get_frequencies_json(train_file, keys)
    freqs.increment(256)  # EOF symbol gets a frequency of 1
    code = freqs.build_code_tree()
    canoncode = huffmancoding.CanonicalCode(tree=code, symbollimit=freqs.get_symbol_limit())
    # Replace code tree with canonical one. For each symbol,
    # the code value may change but the code length stays the same.
    code = canoncode.to_code_tree()
    
    # Read input file again, compress with Huffman coding, and write output file
    with open(test_file, 'r') as fp:
        data_all = fp.readlines()
        data_all = [_l.strip() for _l in data_all]
    pred_data = [_d.split(',')[0] for _d in data_all]
    bit_statistic = BitStatistic(canoncode.codes)
    for line in pred_data:
        for _c in line:
            bit_statistic.step(_c)
    bit_total = bit_statistic.get_result()
    print(bit_total)


class BitStatistic(object):
    def __init__(self, codes):
        self.bit_total = 0
        self.char_total = 0
        self.codes = codes
    def step(self, c):
        self.bit_total += len(self.codes[c])
        self.char_total += 1

    def get_result(self):
        return {'bit':self.bit_total, 'char':self.char_total}

def statistic_json_bit_rate(filepath, keys, codes):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    bit_statistic = BitStatistic(codes)
    traverse_data(data, bit_statistic.step, keys)
    return bit_statistic.get_result()


def get_parser():
    parser = argparse.ArgumentParser("huffman coding for birds/flowers")
    parser.add_argument("--train_json", type=str, default="./EntropyCoding/train.json", help="")
    parser.add_argument("--test_json", type=str, default="./EntropyCoding/test.json", help="")
    parser.add_argument("--mode", choices=['statistic', 'write', 'decompress', 'end_to_end'], type=str, default="statistic", help="")
    parser.add_argument("--keys", type=str, default="data:text", help="")
    parser.add_argument("--output_file", type=str, default="./EntropyCoding/output_file.txt", help="")
    parser.add_argument("--input_file", type=str, default="./EntropyCoding/input_file.txt")
    args = parser.parse_args()
    return args

# Main launcher
if __name__ == "__main__":
    args = get_parser()
    if args.mode == 'statistic':
        main_statistic(args)
    elif args.mode == 'end_to_end':
        main_statistic_end_to_end(args)
    elif args.mode == 'write':
        main_write(args)
    elif args.mode == 'decompress':
        main_decompress(args)
    else:
        raise ValueError