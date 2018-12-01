from typing import List, Optional, Tuple
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import time
from nltk.tokenize import word_tokenize 

from dataset import GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD, GUESSER_TEST_FOLD

MODEL_PATH = "qanta.pt"
UNIQUE_ANSWERS_PATH = "unique_answers.pickle"
WORD2IND_PATH = "word2ind.pickle"

UNK = '<unk>'
PAD = '<pad>'

def load_words(dataset):
    """
    vocabuary building

    Keyword arguments:
    exs: list of input questions-type pairs
    """

    words = set()
    word2ind = {PAD: 0, UNK: 1}
    ind2word = {0: PAD, 1: UNK}
    for example in dataset:
        for sentence in example.sentences:
            for w in sentence.split():
                words.add(w)
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [PAD, UNK] + words
    return words, word2ind, ind2word

def vectorize(ex, word2ind, unique_answers):
    """
    vectorize a single example based on the word2ind dict. 

    Keyword arguments:
    exs: list of input questions-type pairs
    ex: tokenized question sentence (list)
    label: type of question sentence

    Output:  vectorized sentence(python list) and label(int)
    e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
    """
    question_label = list(unique_answers).index(ex.page) if ex.page in unique_answers else 0
    return vectorize_without_label(ex.text, word2ind, unique_answers), question_label

def vectorize_without_label(text, word2ind, unique_answers):
    question_text = word_tokenize(text)
    def toIndex(word):
        return word2ind[word] if word in word2ind.keys() else word2ind[UNK]
    
    return [toIndex(word) for word in question_text]

def extract_unique_answer_list(train_data):
    answer_set = set()
    for example in train_data:
        answer_set.add(example.page)
    return list(answer_set)
    
class Question_Dataset(Dataset):
    """
    Pytorch data class for question classfication data
    """

    def __init__(self, train_data, word2ind, unique_answers):
        self.train_data = train_data
        self.word2ind = word2ind
        self.unique_answers = unique_answers

    def __getitem__(self, index):
        return vectorize(self.train_data[index], self.word2ind, self.unique_answers)
    
    def __len__(self):
        return len(self.train_data)

def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 

    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])
    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch

class DanModel(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_dim=100,
                 n_hidden_units=300, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        self.classifier = nn.Sequential(self.linear1, nn.ReLU(), self.linear2)
        self.softmax = nn.Softmax(1)

    def forward(self, input_text, text_len, is_prob=False):
        embeddings = self.embeddings(input_text)
        logits = self.classifier(embeddings.sum(1)/(text_len.view(embeddings.size(0),-1)))

        return self.softmax(logits) if is_prob else logits

def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set

    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']
        logits = model(question_text, question_len)

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy

class DanGuesser:
    def __init__(self, model=None, unique_answers=None, word2ind=None):
        self.device = torch.device("cpu")
        self.batch_size = 256
        self.num_epochs = 2
        self.grad_clipping = 5
        self.checkpoint = 50
        self.num_workers = 3

        self.model = model
        self.unique_answers = unique_answers
        self.word2ind = word2ind

    def train(self, questions) -> None:
        train_data = questions[GUESSER_TRAIN_FOLD]
        dev_data = questions[GUESSER_DEV_FOLD]
        test_data = questions[GUESSER_TEST_FOLD]

        unique_answers = extract_unique_answer_list(train_data)
        voc, word2ind, ind2word = load_words(train_data)

        with open(UNIQUE_ANSWERS_PATH, "wb") as fp:
            pickle.dump(unique_answers, fp)

        with open(WORD2IND_PATH, "wb") as fp:
            pickle.dump(word2ind, fp)

        model = DanModel(len(unique_answers), len(word2ind))
        model.to(self.device)
        print(model)

        train_dataset = Question_Dataset(train_data, word2ind, unique_answers)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = Question_Dataset(dev_data, word2ind, unique_answers)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=self.batch_size,
                                               sampler=dev_sampler, num_workers=self.num_workers,
                                               collate_fn=batchify)
        accuracy = 0
        for epoch in range(self.num_epochs):
            print('start epoch %d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                               sampler=train_sampler, num_workers=self.num_workers,
                                               collate_fn=batchify)
            accuracy = self.__train_step__(model, train_loader, dev_loader, accuracy)
        print('start testing:\n')

        test_dataset = Question_Dataset(test_data, word2ind, unique_answers)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                               sampler=test_sampler, num_workers=self.num_workers,
                                               collate_fn=batchify)
        evaluate(test_loader, model, self.device)

    def __train_step__(self, model, train_data_loader, dev_data_loader, accuracy):
        """
        Train the current model

        Keyword arguments:
        args: arguments 
        model: model to be trained
        train_data_loader: pytorch build-in data loader output for training examples
        dev_data_loader: pytorch build-in data loader output for dev examples
        accuracy: previous best accuracy
        device: cpu of gpu
        """

        model.train()
        optimizer = torch.optim.Adamax(model.parameters())
        criterion = nn.CrossEntropyLoss()
        print_loss_total = 0
        epoch_loss_total = 0
        start = time.time()

        for idx, batch in enumerate(train_data_loader):
            question_text = batch['text'].to(self.device)
            question_len = batch['len']
            labels = batch['labels']

            out = model(question_text, question_len)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            clip_grad_norm_(model.parameters(), self.grad_clipping)
            print_loss_total += loss.data.numpy()
            epoch_loss_total += loss.data.numpy()

            if idx % self.checkpoint == 0 and idx > 0:
                print_loss_avg = print_loss_total / self.checkpoint

                print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
                print_loss_total = 0
                curr_accuracy = evaluate(dev_data_loader, model, self.device)
                if accuracy < curr_accuracy:
                    torch.save(model, MODEL_PATH)
                    accuracy = curr_accuracy
        return accuracy

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        guesses = []
        for q in questions:
            vectorized_text = vectorize_without_label(q, self.word2ind, self.unique_answers)
            self.model.eval()
            result = self.model(torch.LongTensor([vectorized_text]), torch.FloatTensor([len(vectorized_text)]), is_prob=True)
            top_n, top_i = result.topk(max_n_guesses)
            guesses.append([(self.unique_answers[answer], top_n.flatten()[index].item()) for index, answer in enumerate(top_i.flatten())])

        return guesses

    def save(self):
        return None

    @staticmethod
    def load():
        model = torch.load(MODEL_PATH)
        with open (WORD2IND_PATH, 'rb') as fp:
            word2ind = pickle.load(fp)
        with open (UNIQUE_ANSWERS_PATH, 'rb') as fp:
            unique_answers = pickle.load(fp)
        return DanGuesser(model=model, unique_answers=unique_answers, word2ind=word2ind)
