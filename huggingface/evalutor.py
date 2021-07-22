import  argparse
import os
import sys


sys.path.append(os.path.abspath(os.path.abspath("huggingface")))
from huggingface.HuggingFaceModel import HuggingFaceModel


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path', help='Eval Dataset Path', required=True)
parser.add_argument('-m', '--mod-check', help='Reader Model Checkpoint Path', required=True)
parser.add_argument('-')

def main():
    args = parser.parse_args()
    red = HuggingFaceModel(args.mod_check)
    red.predict_batch(args.data_path, output_to_file=True)