
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
import tensorflow as tf

from soqal import SOQAL
sys.path.append(os.path.abspath("retriever"))
from retriever.TfidfRetriever import HierarchicalTfidf
from retriever.TfidfRetriever import TfidfRetriever
sys.path.append(os.path.abspath(os.path.abspath("huggingface")))
from huggingface.HuggingFaceModel import HuggingFaceModel

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



parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ret-path', help='Retriever Path', required=True)
parser.add_argument('-m', '--mod-check', help='Reader Model Checkpoint Path', required=True)

def main():
    args = parser.parse_args()
    __main__.TfidfRetriever = TfidfRetriever
    base_r = pickle.load(open(args.ret_path, "rb"))
    ret = HierarchicalTfidf(base_r, 1, 1)
    red = HuggingFaceModel(args.mod_check)
    AI = SOQAL(ret, red, 0.999)
    predictions = AI.ask("أفضل لاعب في العالم في كرة القدم")
    print("="*10)
    print("Question answers : ")
    for pred in predictions:
        print("A: ", pred)
    print("="*10)
if __name__ == "__main__":
    main()