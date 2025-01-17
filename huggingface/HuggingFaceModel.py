import json
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import tensorflow as tf

from preprocess import ArabertPreprocessor


model_name = "aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=model_name)

MAX_LENGTH = 384
DOC_STRIDE = 256

output_prediction_file = "predictions.json"

def read_squad_examples(input_file, is_training=False):
    """Read a SQuAD json file into a list of SquadExample."""
    if not isinstance(input_file, list):
        with tf.io.gfile.GFile(input_file) as reader:
            input_data = json.load(reader)["data"]
    else:
        input_data = input_file
    
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

                example = {
                    "id": qas_id,
                    "context": paragraph_text,
                    "question": question_text,
                }
                # (
                #     qas_id=qas_id,
                #     question_text=question_text,
                #     context_text=paragraph_text,
                #     orig_answer_text=orig_answer_text,
                #     start_position=start_position,
                #     end_position=end_position,
                #     is_impossible=False)
                examples.append(example)

    return examples

def substr_with_stride(str, max_len=MAX_LENGTH, stride=DOC_STRIDE):
    start = 0
    end_pos = max_len
    strs = [] 
    while end_pos < len(str):
        strs.append(str[start:end_pos])
        end_pos += stride
        start += stride
    return strs

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))
class HuggingFaceModel:
    def __init__(self, model_checkpoint_path):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)

    def prepare_train_features(self, examples, max_length=MAX_LENGTH, doc_stride=DOC_STRIDE):
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        pad_on_right = self.tokenizer.padding_side == "right"
        tokenized_examples = self.tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    global q,c, off, t_end
                    q = examples["question"][sample_index]
                    c = examples["context"][sample_index]
                    off = offsets
                    t_end = token_end_index
                    while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        pad_on_right = self.tokenizer.padding_side == "right"
        tokenized_examples = self.tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    def query_model(self, question, context):
        inputs = self.tokenizer(question, context,
                                max_length=MAX_LENGTH,
                                stride=DOC_STRIDE,
                                return_overflowing_tokens=True,
                                padding="max_length",
                                truncation="only_second",
                                # return_offsets_mapping=True,
                                return_tensors="pt")
        inputs.pop("overflow_to_sample_mapping")
        input_ids = inputs["input_ids"].tolist()
        outputs = self.model(**inputs)
        answers = []
        for i, output in enumerate(outputs.start_logits):
            answer_start_scores = output
            answer_end_scores = outputs.end_logits[i]

            answer_start_logit = torch.max(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end_logit = torch.max(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[i][answer_start:answer_end]))
            answers.append({"s": answer_start_logit, "e": answer_end_logit, "t": answer}) 
        return answers
    
    def predict_batch(self, examples, output_to_file=False, out_path=None):
        print("eval before1")

        if output_to_file:
            all_predictions = {}
        print("eval before")
        eval_examples = read_squad_examples(examples)
        nbest = {}
        for example in eval_examples[:100]:
            context  = example["context"]
            question = example["question"]
            answers = self.query_model(question, context)
            for answer in answers:
                if example["id"] in nbest:
                    old_answer_score = nbest[example["id"]][0]["start_logit"] * nbest[example["id"]][0]["end_logit"]
                    new_answer_score = answer["s"] * answer["e"]
                    if new_answer_score > old_answer_score:
                        nbest[example["id"]][0] = {                  # just for getting along with SOQAL interface
                            'start_logit': answer["s"],
                            'end_logit': answer["e"],
                            'text': answer["t"]
                        }
                else:
                    nbest[example["id"]] = {}
                    nbest[example["id"]][0] = {                  # just for getting along with SOQAL interface
                        'start_logit': answer["s"],
                        'end_logit': answer["e"],
                        'text': answer["t"]
                    }
            if output_to_file:
                all_predictions[example["id"]] = nbest[example["id"]][0]["text"]
        if output_to_file:
            with tf.gfile.GFile(out_path, "w") as writer:
                    writer.write(json.dumps(all_predictions, indent=4) + "\n")
        return nbest