import  argparse
import os
import sys


from HuggingFaceModel import HuggingFaceModel

print("Hereeee")
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path', help='Eval Dataset Path', required=True)
parser.add_argument('-m', '--mod-check', help='Reader Model Checkpoint Path', required=True)
parser.add_argument('-o', '--out-path', help='Out path', required=True)

def main():
    args = parser.parse_args()
    print("main before")

    red = HuggingFaceModel(args.mod_check)
    red.predict_batch(args.data_path, output_to_file=True, out_path=args.out_path)

if __name__ == "__main__":
    main()