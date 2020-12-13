import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
data_directory = "../Data2/"
data_file = "Data_Tunes.txt"
charIndex_json = "char_to_index.json"
model_weights_directory = '../Data2/Model_Weights/'
BATCH_SIZE = 16
SEQ_LENGTH = 64
def read_batches(all_chars, unique_chars):
    length = all_chars.shape[0]
    batch_chars = int(length / BATCH_SIZE) 
    
    for start in range(0, batch_chars - SEQ_LENGTH, 64): 
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))    
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, unique_chars))  
        for batch_index in range(0, 16):   
            for i in range(0, 64): 
                X[batch_index, i] = all_chars[batch_index * batch_chars + start + i]
                Y[batch_index, i, all_chars[batch_index * batch_chars + start + i + 1]] = 1
        yield X, Y
def built_model(batch_size, seq_length, unique_chars):
    model = Sequential()
    
    model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (batch_size, seq_length), name = "embd_1")) 
    
    model.add(LSTM(256, return_sequences = True, stateful = True, name = "lstm_first"))
    model.add(Dropout(0.2, name = "drp_1"))
    
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(TimeDistributed(Dense(unique_chars)))
    model.add(Activation("softmax"))
    
    model.load_weights("../Data/Model_Weights/Weights_80.h5", by_name = True)
    
    return model
def training_model(data, epochs = 90):
    char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
    print("Number of unique characters in our whole tunes database = {}".format(len(char_to_index))) 
    
    with open(os.path.join(data_directory, charIndex_json), mode = "w") as f:
        json.dump(char_to_index, f)
        
    index_to_char = {i: ch for (ch, i) in char_to_index.items()}
    unique_chars = len(char_to_index)
    
    model = built_model(BATCH_SIZE, SEQ_LENGTH, unique_chars)
    model.summary()
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    all_characters = np.asarray([char_to_index[c] for c in data], dtype = np.int32)
    print("Total number of characters = "+str(all_characters.shape[0])) 
    
    epoch_number, loss, accuracy = [], [], []
    
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        final_epoch_loss, final_epoch_accuracy = 0, 0
        epoch_number.append(epoch+1)
        
        for i, (x, y) in enumerate(read_batches(all_characters, unique_chars)):
            final_epoch_loss, final_epoch_accuracy = model.train_on_batch(x, y)
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_loss, final_epoch_accuracy))
        loss.append(final_epoch_loss)
        accuracy.append(final_epoch_accuracy)
        
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(model_weights_directory):
                os.makedirs(model_weights_directory)
            model.save_weights(os.path.join(model_weights_directory, "Weights_{}.h5".format(epoch+1)))
            print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch+1, epoch+1))
    
    log_frame = pd.DataFrame(columns = ["Epoch", "Loss", "Accuracy"])
    log_frame["Epoch"] = epoch_number
    log_frame["Loss"] = loss
    log_frame["Accuracy"] = accuracy
    log_frame.to_csv("../Data2/log.csv", index = False)
file = open(os.path.join(data_directory, data_file), mode = 'r')
data = file.read()
file.close()
if __name__ == "__main__":
    training_model(data)