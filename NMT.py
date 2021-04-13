from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Citation:
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://www.programcreek.com/python/example/100048/nltk.translate.bleu_score.SmoothingFunction


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''Loading data files'''

SOS_token = 0
EOS_token = 1


# use Lang class to keep track of the unique index of every word
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# for simplicity, we convert Unicode characters to ASCII,
# convert everything to lowercase, and trim most punctuation
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    return s


# Read the data file, divide the file into several lines, and then divide the lines into pairs
def readLangs(lang1, lang2):
    print("Reading lines...")

    # Read the file and split into lines
    lines_train_en = open('data/train_en.txt', encoding='utf-8').read().strip().split('\n')
    lines_train_vi = open('data/train_vi.txt', encoding='utf-8').read().strip().split('\n')
    lines_test_en = open('data/test_en.txt', encoding='utf-8').read().strip().split('\n')
    lines_test_vi = open('data/test_vi.txt', encoding='utf-8').read().strip().split('\n')

    # split every line into pairs and normalize
    pairs_train = [[normalizeString(lines_train_en[i]), normalizeString(lines_train_vi[i])] for i in range(len(lines_train_en))]
    pairs_test = [[normalizeString(lines_test_en[i]), normalizeString(lines_test_vi[i])] for i in range(len(lines_test_en))]

    # make Lang instances，translate: English -> Vietnamese
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs_train, pairs_test


# define max length for filtering
MAX_LENGTH = 50  # original 40


# trim the dataset to only relatively short and simple sentences
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# the complete process of preparing the data
def prepareData(lang1, lang2):
    # read text file and split into lines, split lines into pairs, and normalize
    input_lang, output_lang, pairs_train, pairs_test = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs_train))
    print("Read %s sentence pairs" % len(pairs_test))

    # filter by length
    pairs_train = filterPairs(pairs_train)
    pairs_test = filterPairs(pairs_test)
    print("Trimmed to %s training sentence pairs" % len(pairs_train))
    print("Trimmed to %s testing sentence pairs" % len(pairs_test))
    print("Counting words...")

    # create a list of words in a sentence in pairs
    for pair in pairs_train:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    for pair in pairs_test:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    # display the number of word pairs
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs_train, pairs_test


'''Seq2Seq model'''


# Encoder
# For each input word, the encoder outputs a vector and a hidden state,
# and uses the hidden state for the next input word
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Attention Decoder
# Use encoder to output vector and output word sequence to create translation
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


'''Training'''


# Preparing Training Data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])  # input_tensor: the index of the word in the input sentence
    target_tensor = tensorFromSentence(output_lang, pair[1])  # target_tensor: the index of the word in the target sentence
    return input_tensor, target_tensor


# Training the Model
teacher_forcing_ratio = 0.5  # using teacher forcing leads to converge faster


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(pairs_train, encoder, decoder, n_iters, print_every=2000, learning_rate=0.01):
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)  # Initialize optimizers
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print("Training...")

    for iter in range(1, n_iters + 1):
        training_pair = pairs_train[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:  # calculate and display loss
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Loss %d: %.6f' % (iter, print_loss_avg))


# start training process and save training model
def start_train(encoder1, attn_decoder1, input_lang, output_lang, pairs):
    train_iteration = 35000

    train_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(train_iteration)]

    trainIters(train_pairs, encoder1, attn_decoder1, n_iters=train_iteration, print_every=2000)

    # save the training model
    print('Finished training')
    torch.save(encoder1.state_dict(), './model/encoderRNN.pth')
    torch.save(attn_decoder1.state_dict(), './model/attnDecoderRNN.pth')


'''Evaluation'''


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length = MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


'''Testing'''


def start_test(encoder, attn_decoder, input_lang, output_lang, pairs):
    bleu_score = calculate_Bleu(encoder, attn_decoder, pairs, input_lang, output_lang)  # calculate Bleu score
    print("Average BLEU: " + str(bleu_score))


def calculate_Bleu(encoder, decoder, pairs, input_lang, output_lang):
    bleu_score = []  # initialize list, which is used to store every bleu score

    for pair in pairs:
        try:
            output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        except RuntimeError:
            pass
        output_sentence = ' '.join(output_words)
        # calculate bleu score for each pair
        bleu_score.append(sentence_bleu([output_sentence], pair[1], smoothing_function=SmoothingFunction().method1))

    bleu_mean = np.mean(bleu_score)  # calculate the average of all bleu scores
    return bleu_mean


'''get the keyword ('train', 'test' or 'translation') from the console'''


input_lang, output_lang, train_pairs, test_pairs = prepareData('En', 'Vi')

hidden_size = 512  # has 512 hidden nodes
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

if len(sys.argv) == 0 or sys.argv[1] == "train":
    start_train(encoder, attn_decoder, input_lang, output_lang, train_pairs)

elif sys.argv[1] == "test":
    print('Loading model...')
    encoder.load_state_dict(torch.load('./model/encoderRNN.pth'))
    attn_decoder.load_state_dict(torch.load('./model/attnDecoderRNN.pth'))

    start_test(encoder, attn_decoder, input_lang, output_lang, test_pairs)

elif sys.argv[1] == "translate":
    print('Loading model...')
    encoder.load_state_dict(torch.load('./model/encoderRNN.pth'))
    attn_decoder.load_state_dict(torch.load('./model/attnDecoderRNN.pth'))

    while True:
        print("Note: You need to enter \"ctrl + C\" to end the translation process.")
        original_text = input("> ")
        original_text = str(original_text)  # transform the input into string
        original_text = unicodeToAscii(original_text.lower().strip())  # normalize
        decoded_words, _ = evaluate(encoder, attn_decoder, original_text, input_lang, output_lang)
        translation = ' '.join(decoded_words)
        print(translation)
