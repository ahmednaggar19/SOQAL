
import pickle
import  argparse
import threading
import json
from time import sleep
import os
import bottle
from bottle import static_file
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import sys
import __main__
import tf

from soqal import SOQAL
sys.path.append(os.path.abspath("retriever"))
from retriever.TfidfRetriever import HierarchicalTfidf
from retriever.TfidfRetriever import TfidfRetriever
sys.path.append(os.path.abspath("bert"))
from bert.Bert_model import SquadExample

app = bottle.Bottle()
query = []
response = ""
my_module = os.path.abspath(__file__)
parent_dir = os.path.dirname(my_module)
static_dir = os.path.join(parent_dir, 'static')

@app.get("/")
def home():
    with open('demo_open.html', encoding='utf-8') as fl:
        html = fl.read()
        return html

@app.get('/static/<filename>')
def server_static(filename):
    return static_file(filename, root=static_dir)

@app.post('/answer')
def answer():
    question = bottle.request.json['question']
    print("received question: {}".format(question))
    # if not passage or not question:
    #     exit()
    global query, response
    query = question
    if query != "":
        while not response:
            sleep(0.1)
    else:
        response = "Please write a question"
    print("received response: {}".format(response))
    response_ = {"answer": response} 
    response = []
    return response_


class Demo(object):
    def __init__(self, model, config):
        self.model = model
        run_event = threading.Event()
        run_event.set()
        self.close_thread = True
        threading.Thread(target=self.demo_backend).start()
        app.run(port=9999, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            self.close_thread = False
    def demo_backend(self):
        global query, response
        while self.close_thread:
            sleep(0.1)
            if query:
                response = self.model.ask(query)
                query = []

def read_squad_examples(input_file):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.io.gfile.GFile(input_file) as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=paragraph_text,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=False)
                examples.append(example)

    return examples


class HuggingFaceModel:
    def __init__(self, model_checkpoint_path):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)

    def predict_batch(self, input_data):
        eval_examples = read_squad_examples(input_data)
        nbest = {}
        idx = 0
        for example in eval_examples:
            # TODO:  extract question and text from example
            inputs = self.tokenizer(example.question_text, example.context_text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
            outputs = self.model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            answer_start = torch.argmax(
                answer_start_scores
            )  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            nbest[idx][0] = {
                'start_logit': outputs.start_logits[answer_start],
                'end_logits': outputs.end_logits[answer_end],
                'text': answer
            }
            idx += 1

        return nbest
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ret-path', help='Retriever Path', required=True)
parser.add_argument('-m', '--mod-check', help='Reader Model Checkpoint Path', required=True)

def main():
    args = parser.parse_args()
    __main__.TfidfRetriever = TfidfRetriever
    base_r = pickle.load(open(args.ret_path, "rb"))
    ret = HierarchicalTfidf(base_r, 50, 50)
    red = HuggingFaceModel(args.mod_check)
    AI = SOQAL(ret, red, 0.999)
    pred = AI.ask("من بطل كأس العالم ١٩٩٨؟")
    print(pred)

if __name__ == "__main__":
    main()