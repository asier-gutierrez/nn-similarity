import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def define_alphabet():
    base_en = 'abcdefghijklmnopqrstuvwxyz'
    special_chars = ' !?¿¡'
    german = 'äöüß'
    italian = 'àèéìíòóùú'
    french = 'àâæçéèêêîïôœùûüÿ'
    spanish = 'áéíóúüñ'
    czech = 'áčďéěíjňóřšťúůýž'
    slovak = 'áäčďdzdžéíĺľňóôŕšťúýž'
    all_lang_chars = base_en + german + italian + french + spanish + czech + slovak
    small_chars = list(set(list(all_lang_chars)))
    small_chars.sort()
    big_chars = list(set(list(all_lang_chars.upper())))
    big_chars.sort()
    small_chars += special_chars
    letters_string = ''
    letters = small_chars + big_chars
    for letter in letters:
        letters_string += letter
    return small_chars, big_chars, letters_string


def get_sample_text(file_content, start_index, sample_size):
    # we want to start from full first word
    # if the firts character is not space, move to next ones
    while not (file_content[start_index].isspace()):
        start_index += 1
    # now we look for first non-space character - beginning of any word
    while file_content[start_index].isspace():
        start_index += 1
    end_index = start_index + sample_size
    # we also want full words at the end
    while not (file_content[end_index].isspace()):
        end_index -= 1
    return file_content[start_index:end_index]


def count_chars(text, alphabet):
    alphabet_counts = []
    for letter in alphabet:
        count = text.count(letter)
        alphabet_counts.append(count)
    return alphabet_counts


def get_input_row(content, start_index, sample_size, alphabet):
    sample_text = get_sample_text(content, start_index, sample_size)
    counted_chars_all = count_chars(sample_text.lower(), alphabet[0])
    counted_chars_big = count_chars(sample_text, alphabet[1])
    all_parts = counted_chars_all + counted_chars_big
    return all_parts


def train(num_classes, X_train, Y_train, X_test, Y_test):
    model = tf.keras.Sequential()
    model.add(
        Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    x_train, y_train, x_test, y_test = X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    train_idxs = np.random.permutation(len(x_train))
    test_idxs = np.random.permutation(len(x_test))

    # Train randomization
    x_train = x_train[train_idxs]
    y_train = y_train[train_idxs]

    # Test randomization
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=20,
                        verbose=1,
                        validation_split=0.1)
    return model


def prepare():
    LANGUAGES_DICT = {'en': 0, 'fr': 1, 'es': 2, 'it': 3, 'de': 4, 'sk': 5, 'cs': 6}
    num_classes = len(LANGUAGES_DICT)

    # Length of cleaned text used for training and prediction - 140 chars
    MAX_LEN = 140

    # number of language samples per language that we will extract from source files
    NUM_SAMPLES = 250000

    # For reproducibility
    SEED = 42

    # Load the Alphabet
    alphabet = define_alphabet()
    print('ALPHABET:')
    print(alphabet[2])

    VOCAB_SIZE = len(alphabet[2])
    print('ALPHABET LEN(VOCAB SIZE):', VOCAB_SIZE)

    # Folders from where load / store the raw, source, cleaned, samples and train_test data
    data_directory = "./data/language_identification"
    source_directory = os.path.join(data_directory, 'source')
    cleaned_directory = os.path.join(data_directory, 'cleaned')
    samples_directory = os.path.join(data_directory, 'samples')
    train_test_directory = os.path.join(data_directory, 'train_test')

    path = os.path.join(cleaned_directory, "de_cleaned.txt")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        random_index = random.randrange(0, len(content) - 2 * MAX_LEN)
        sample_text = get_sample_text(content, random_index, MAX_LEN)
        print("1. SAMPLE TEXT: \n", sample_text)
        print("\n2. REFERENCE ALPHABET: \n", alphabet[0] + alphabet[1])

        sample_input_row = get_input_row(content, random_index, MAX_LEN, alphabet)
        print("\n3. SAMPLE INPUT ROW: \n", sample_input_row)

        input_size = len(sample_input_row)
        if input_size != VOCAB_SIZE:
            print("Something strange happened!")

        print("\n4. INPUT SIZE (VOCAB SIZE): ", input_size)
        del content

    def size_mb(size):
        size_mb = '{:.2f}'.format(size / (1000 * 1000.0))
        return size_mb + " MB"

    # Now we have preprocessing utility functions ready. Let's use them to process each cleaned language file
    # and turn text data into numerical data samples for our neural network
    # prepare numpy array
    sample_data = np.empty((NUM_SAMPLES * len(LANGUAGES_DICT), input_size + 1), dtype=np.uint16)
    lang_seq = 0  # offset for each language data
    jump_reduce = 0.2  # part of characters removed from jump to avoid passing the end of file

    for lang_code in LANGUAGES_DICT:
        start_index = 0
        path = os.path.join(cleaned_directory, lang_code + "_cleaned.txt")
        with open(path, 'r', encoding='utf-8') as f:
            print("Processing file : " + path)
            file_content = f.read()
            content_length = len(file_content)
            remaining = content_length - MAX_LEN * NUM_SAMPLES
            jump = int(((remaining / NUM_SAMPLES) * 3) / 4)
            print("File size : ", size_mb(content_length), \
                  " | # possible samples : ", int(content_length / VOCAB_SIZE), \
                  "| # skip chars : " + str(jump))
            for idx in range(NUM_SAMPLES):
                input_row = get_input_row(file_content, start_index, MAX_LEN, alphabet)
                sample_data[NUM_SAMPLES * lang_seq + idx,] = input_row + [LANGUAGES_DICT[lang_code]]
                start_index += MAX_LEN + jump
            del file_content
        lang_seq += 1
        print(100 * "-")

    # Let's randomy shuffle the data
    np.random.shuffle(sample_data)
    # reference input size
    print("Vocab Size : ", VOCAB_SIZE)
    print(100 * "-")
    print("Samples array size : ", sample_data.shape)

    # Create the the sample dirctory if not exists
    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    # Save compressed sample data to disk
    path_smpl = os.path.join(samples_directory, "lang_samples_" + str(VOCAB_SIZE) + ".npz")
    np.savez_compressed(path_smpl, data=sample_data)
    print(path_smpl, "size : ", size_mb(os.path.getsize(path_smpl)))
    del sample_data

    # utility function to turn language id into language code
    def decode_langid(langid):
        for dname, did in LANGUAGES_DICT.items():
            if did == langid:
                return dname

    # Loading the data
    path_smpl = os.path.join(samples_directory, "lang_samples_" + str(VOCAB_SIZE) + ".npz")
    dt = np.load(path_smpl)['data']

    # Sanity chech on a random sample
    random_index = random.randrange(0, dt.shape[0])
    print("Sample record : \n", dt[random_index,])
    print("\nSample language : ", decode_langid(dt[random_index,][VOCAB_SIZE]))

    # Check if the data have equal share of different languages
    print("\nDataset shape (Total_samples, Alphabet):", dt.shape)
    bins = np.bincount(dt[:, input_size])

    print("Language bins count (samples per language): ")
    for lang_code in LANGUAGES_DICT:
        print(lang_code, bins[LANGUAGES_DICT[lang_code]])

    # we need to preprocess data for DNN yet again - scale it
    # scaling will ensure that our optimization algorithm (variation of gradient descent) will converge well
    # we need also ensure one-hot econding of target classes for softmax output layer
    # let's convert datatype before processing to float
    dt = dt.astype(np.float32)
    # X and Y split
    X = dt[:, 0:input_size]  # Samples
    Y = dt[:, input_size]  # The last element is the label
    del dt

    # Random index to check random sample
    random_index = random.randrange(0, X.shape[0])
    print("Example data before processing:")
    print("X : \n", X[random_index,])
    print("Y : \n", Y[random_index])

    # X PREPROCESSING
    # Feature Standardization - Standar scaler will be useful later during DNN prediction
    standard_scaler = preprocessing.StandardScaler().fit(X)
    X = standard_scaler.transform(X)
    print("X preprocessed shape :", X.shape)

    # Y PREPROCESSINGY
    # One-hot encoding
    Y = tf.keras.utils.to_categorical(Y, num_classes=len(LANGUAGES_DICT))

    # See the sample data
    print("\nExample data after processing:")
    print("X : \n", X[random_index,])
    print("Y : \n", Y[random_index])

    # Train/test split. Static seed to have comparable results for different runs
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=SEED)
    del X, Y

    # Create the train / test directory if not extists
    if not os.path.exists(train_test_directory):
        os.makedirs(train_test_directory)

    # Save compressed train_test data to disk
    path_tt = os.path.join(train_test_directory, "train_test_data_" + str(VOCAB_SIZE) + ".npz")
    np.savez_compressed(path_tt, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print(path_tt, "size : ", size_mb(os.path.getsize(path_tt)))
    del X_train, Y_train, X_test, Y_test

    # Load train data first from file
    path_tt = os.path.join(train_test_directory, "train_test_data_" + str(VOCAB_SIZE) + ".npz")
    train_test_data = np.load(path_tt)

    # Train Set
    X_train = train_test_data['X_train']
    print("X_train: ", X_train.shape)
    Y_train = train_test_data['Y_train']
    print("Y_train: ", Y_train.shape)

    # Test Set
    X_test = train_test_data['X_test']
    print("X_test: ", X_test.shape)
    Y_test = train_test_data['Y_test']
    print("Y_test: ", Y_test.shape)

    del train_test_data
    return num_classes, X_train, Y_train, X_test, Y_test