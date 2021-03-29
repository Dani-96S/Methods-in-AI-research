from plot_utils import repeat_plot
from data_retrieval import return_speechacts
from utils import norm_input, closest_word
from keras.layers import LSTM, Dense, Embedding, Activation, Dropout
from keras.models import Sequential, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class SpeechActModel:

    def __init__(self, retrain=True,
                 sentence_size=15,
                 v_split=0.1,
                 n_epochs=5,
                 embd_size=265,
                 lstm_size=64,
                 den1_size=0,
                 drop_rate=0.5,
                 den2_size=0,
                 activation="softmax",
                 optimizer="adam",
                 loss_func="categorical_crossentropy"):
        """
        param retrain: Retrain the model if True, else read from file
        """
        self.sentence_size = sentence_size
        self.v_split = v_split
        self.n_epochs = n_epochs
        self.embd_size = embd_size
        self.lstm_size = lstm_size
        self.den1_size = den1_size
        self.drop_rate = drop_rate
        self.den2_size = den2_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.prepare()
        if retrain:
            self.model = self.fit_model()
        else:
            self.model = self.load_model()

    def prepare(self):
        """
        Take the data from the speechact files
        Prepare the data for use with a word-embedding layer
        defines:
        original data:  x_train, x_test, y_train, y_test
        tokenized data: xt_train, xt_test, yt_train, yt_test
        tokenizers:     tokenizer_x, tokenizer_y
        """
        # get data from file
        train_data, test_data = return_speechacts()
        # y are the speechacts or 'labels'
        y_train = [t.split(' ')[0] for t in train_data]
        y_test = [t.split(' ')[0] for t in test_data]
        # x are the sentences
        x_train = [" ".join(t.split(' ')[1:]) for t in train_data]
        x_test = [" ".join(t.split(' ')[1:]) for t in test_data]
        # use the tokenizer and padding from keras to assign arrays of integers
        # to sentences, out of vocabulary token is 1
        self.tokenizer_x = Tokenizer(oov_token=1)
        self.tokenizer_x.fit_on_texts(x_train + x_test)
        xt_train = self.tokenizer_x.texts_to_sequences(x_train)
        xt_train = pad_sequences(xt_train, maxlen=self.sentence_size,
                                 dtype='int32')
        xt_test = self.tokenizer_x.texts_to_sequences(x_test)
        xt_test = pad_sequences(xt_test, maxlen=self.sentence_size,
                                dtype='int32')
        # vocab is the number of words in our vocabulary
        self.vocab = len(self.tokenizer_x.word_index) + 1
        # do the same for labels
        self.tokenizer_y = Tokenizer()
        self.tokenizer_y.fit_on_texts(y_train + y_test)
        yt_train = self.tokenizer_y.texts_to_sequences(y_train)
        yt_train = [t[0] for t in yt_train]
        yt_train = to_categorical(yt_train)
        yt_test = self.tokenizer_y.texts_to_sequences(y_test)
        yt_test = [t[0] for t in yt_test]
        yt_test = to_categorical(yt_test)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.xt_train = xt_train
        self.yt_train = yt_train
        self.xt_test = xt_test
        self.yt_test = yt_test

    def make_model(self):
        """
        Defines the parameters and layers of the model.
        """
        model = Sequential()
        model.add(Embedding(self.vocab, self.embd_size,
                            input_length=self.sentence_size))
        model.add(LSTM(self.lstm_size, return_sequences=False))
        if self.den1_size > 0:
            model.add(Dense(self.den1_size, activation='relu'))
        if self.drop_rate > 0:
            model.add(Dropout(self.drop_rate))
        if self.den2_size > 0:
            model.add(Dense(self.den2_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Activation(self.activation))
        model.compile(optimizer=self.optimizer,
                      loss=self.loss_func,
                      metrics=['accuracy'])
        return model

    def fit_model(self):
        """
        Trains the model on the data in xt_train with labels in yt_train
        """
        model = self.make_model()
        self.history = model.fit(x=self.xt_train, y=self.yt_train,
                                 epochs=self.n_epochs, verbose=0,
                                 validation_split=self.v_split, shuffle=True)
        self.eval_model(model)
        self.save_model(model)
        return model

    def eval_model(self, model):
        """
        Tests the actual loss and accuracy of the model
        using xt_test and xy_test
        """
        evaluation = model.evaluate(x=self.xt_test, y=self.yt_test)
        print("loss    : " + str(round(evaluation[0]*100, 2)) + "%")
        print("accuracy: " + str(round(evaluation[1]*100, 2)) + "%")

    @staticmethod
    def save_model(model, filename="model.json"):
        """
        Save the trained model with weights in a h5 file
        """
        model_json = model.to_json()
        with open(filename, "w") as json_file:
            json_file.write(model_json)
            model.save_weights("model.h5")
        print("Saved model to disk")

    @staticmethod
    def load_model(filename="model.json"):
        """
        Loads the trained model with weights from a h5 file
        """
        with open(filename, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights("model.h5")
        return model

    def sentence_prediction(self, sentence, echo=False):
        """
        Takes a sentence and predicts the associated speech act
        param echo: print the predictions to the console
        returns a list of tuples [(probability::Double, speechact::String)]
        """
        sentence = norm_input(sentence)
        sentence = self.typing_errors(sentence)
        sent_seq = self.tokenizer_x.texts_to_sequences([sentence])
        sentence_tok = pad_sequences(sent_seq, maxlen=self.sentence_size,
                                     dtype='int32')
        wi = self.tokenizer_y.word_index
        pr = self.model.predict(sentence_tok)[0]
        predictions = []
        for speechact, index in wi.items():
            predictions.append((round(pr[index]*100, 2), speechact))
        predictions.sort(key=lambda tup: tup[0])
        if echo:
            for p in predictions:
                print(str(p[0]).rjust(7) + " -> " + p[1])
        return predictions[-1][1]

    def typing_errors(self, sentence):
        corrected = []
        sentence = sentence.split(" ")
        lex = [word for word, _ in self.tokenizer_x.word_index.items()]
        for word in sentence:
            best_match = closest_word(word, lex, treshold=0.8)
            if best_match:
                corrected.append(best_match)
            else:
                corrected.append(word)
        return " ".join(corrected)


if __name__ == "__main__":
    retrain = True
    plot = False
    user = True

    s = SpeechActModel(retrain=retrain,
                       n_epochs=5,
                       embd_size=256,
                       lstm_size=64,
                       drop_rate=0.5)
    while user:
        user_input = input("Please enter a question, [q] to quit:\n")
        if user_input == "q":
            break
        s.sentence_prediction(user_input, echo=True)
    if plot:
        repeat_plot(s, 5)


def the_big_test(r):
    print("\n\n=== Model 1 ===\n")
    s = SpeechActModel(n_epochs=5,
                       embd_size=64,
                       lstm_size=64)
    repeat_plot(s, r)
    print("\n\n=== Model 2 ===\n")
    s = SpeechActModel(n_epochs=5,
                       embd_size=265,
                       lstm_size=64)
    repeat_plot(s, r)
    print("\n\n=== Model 3 ===\n")
    s = SpeechActModel(n_epochs=5,
                       embd_size=16,
                       lstm_size=32)
    repeat_plot(s, r)
    print("\n\n=== Model 4 ===\n")
    s = SpeechActModel(n_epochs=20,
                       embd_size=64,
                       lstm_size=256,
                       den1_size=128)
    repeat_plot(s, r)
    print("\n\n=== Model 5 ===\n")
    s = SpeechActModel(n_epochs=5,
                       embd_size=64,
                       lstm_size=32,
                       den1_size=32)
    repeat_plot(s, r)
    print("\n\n=== Model 6 ===\n")
    s = SpeechActModel(n_epochs=5,
                       embd_size=64,
                       lstm_size=32,
                       den1_size=32,
                       drop_rate=0.25)
    repeat_plot(s, r)
    print("\n\n=== Model 7 ===\n")
    s = SpeechActModel(n_epochs=5,
                       embd_size=64,
                       lstm_size=32,
                       den1_size=32,
                       drop_rate=0.5)
    repeat_plot(s, r)
    print("\n\n=== Model 8 ===\n")
    s = SpeechActModel(n_epochs=5,
                       embd_size=512,
                       lstm_size=64,
                       drop_rate=0.5)
    repeat_plot(s, r)
