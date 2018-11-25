from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

import util
from dataset import QuizBowlDataset

import re
import nltk
from nltk.corpus import stopwords
import nltk.tokenize as nt
from pattern.en import singularize
# from apiclient.discovery import build
import pandas as pd

MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs


class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    # Informational function
    def count_found_answers(self, answers, wiki_to_type):
        count_found = 0
        for i, answer in enumerate(answers):
            if i % 25 == 0: print("Iteration: " + str(i))
            if wiki_to_type.index[wiki_to_type["Entity"] == answer.replace('_', ' ')].size:
                count_found += 1
        print("Count found: " + str(count_found))
        print("Total answers: " + str(len(answers)))

    # Informational function
    def rank_unique_answers(self, answers):
        answer_dict = defaultdict(int)
        for answer in answers:
            answer_dict[answer] += 1
        sorted_answer_dict = sorted(answer_dict.items(), key=lambda kv: kv[1], reverse=True)
        print("Num unique answers: " + str(len(answer_dict.keys())))
        with open('sorted_answer_dict.json', 'w') as file:
            file.write(json.dumps(sorted_answer_dict))

    def train_type_filter(self, training_data):
        wiki_to_type = pd.read_table("slim-freebase-types.tsv", engine="python")
        stop_words = set(stopwords.words('english'))
        questions = training_data[0][:1000]
        answers = training_data[1][:1000]
        type_dict = defaultdict(lambda: {'word_count': 0, 'answer_type_set': set()})
        for i, q in enumerate(questions):
            if i % 100 == 0:
                print("Starting iteration " + str(i))
            sentence_tokens = [nt.word_tokenize(sentence) for sentence in q]
            pos_sentences = [nltk.pos_tag(s) for s in sentence_tokens]
            for sentence in pos_sentences:
                add_word = False
                for token in sentence:
                    word = token[0]
                    pos = token[1]
                    if add_word:
                        if word not in stop_words and pos in ["NN", "NNS"]:
                            type_dict[singularize(token[0])]["word_count"] += 1
                            type_dict[singularize(token[0])]["answer_type_set"].update(self.get_answer_type(answers[i], wiki_to_type))
                            add_word = False
                    if word.lower() in ["this", "these"]:
                        add_word = True

        sorted_type_dict = sorted(type_dict.items(), key=lambda kv: kv[1]["word_count"], reverse=True)
        with open('sorted_type_dict.json', 'w') as file:
            file.write(json.dumps(sorted_type_dict))
        return None

    def get_answer_type(self, answer, wiki_to_type):
        answer_indices = wiki_to_type.index[wiki_to_type["Entity"] == answer.replace('_', ' ')]
        return wiki_to_type.iloc[answer_indices]["Type"].values

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


def create_app(enable_batch=True):
    tfidf_guesser = TfidfGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(tfidf_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(dataset.training_data())
    tfidf_guesser.save()

def train_type_filter():
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train_type_filter(dataset.training_data())


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
