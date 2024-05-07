from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
import data 
import tensorflow as tf
import numpy as np
import os
from keras.layers import TimeDistributed
from keras.layers import LSTMCell, RNN, LSTM

FILE_PATH_WEIGHT_BTSNN = "WeightBetsoftwareAIV1.h5"

class BTSPredictNeuralNetwork:
    def __init__(self, number_parameter) -> None:
        self.model = self.create_model(number_parameter)
    
    def save_weight(self):
        """Save the weights of the neural network
        """
        print("Sauvegarde des poids du modèle en cours, ne fermez pas !")
        self.model.save_weights(FILE_PATH_WEIGHT_BTSNN)
        print("Sauvegarde effectué")
        return
    
    def load_weight(self):
        self.model.load_weights(FILE_PATH_WEIGHT_BTSNN)
        
    def create_model(self, number_parameter: int):
        """
        Creates a sequential neural network model with improved architecture and hyperparameters.

        Args:
            number_parameter (int): Number of input features.

        Returns:
            tf.keras.Sequential: Compiled model ready for training.
        """
        input_shape = (4, 10, 21)  # (num_lists, num_matches, num_stats)

        # Define the LSTM layer
        lstm_layer = tf.keras.layers.LSTM(units=16, return_sequences=True)

        # Define the input layer
        input_layer = tf.keras.layers.Input(shape=input_shape)

        # Process each list of matches separately
        lstm_outputs = []
        for i in range(input_shape[0]):
            lstm_output = lstm_layer(input_layer[:, i, :, :])
            lstm_outputs.append(lstm_output)

        # Concatenate the outputs
        concatenated_output = tf.keras.layers.Concatenate()(lstm_outputs)
        
        x = tf.keras.layers.Dense(256, activation='relu')(concatenated_output[:, 0, :])
        
        for _ in range(50):
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        
        
        # Define the output layer
        output_layer = tf.keras.layers.Dense(2, activation='softmax')(x)

        # Define the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


        return model
    

    def train(self, x_training, y_training):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(x_training, y_training, verbose=1, epochs=100, batch_size=256, validation_split=0.2, use_multiprocessing=False, callbacks=[early_stopping])
        
    def predict(self, x):
        
        return self.model.predict(x=x, verbose=0)
    
    def test_this_league(self, league_str):
        for league in data.get_all_league():
            if league.ID_League == league_str:
                league = league
                break
            
        list_matchs = league.get_list_matches()
        true_prediction, score, compteur_prediction = 0, 0, 0
        
        for match in list_matchs:
            have_good_prediction_bool = self.is_nn_have_good_predict(match, matches_list_league=list_matchs)
            if have_good_prediction_bool is None:
                continue
            compteur_prediction += 1
            score -= 1
            if have_good_prediction_bool:
                if match.get_bts() == 1:
                    true_prediction += 1
                    score += match.bts_yes_odd
                else:
                    true_prediction += 1
                    score += match.bts_no_odd
        
        print(f"Test de la league {league.get_league_name()} pour la saison {league.get_season()} :")
        print(f"Pourcentage de prédiction juste : {true_prediction/compteur_prediction*100:.2f}%")
        print(f"Score : {score:.2f}")
        return score
    
    def is_nn_have_good_predict(self, match, matches_list_league=[]):
        x, y_true = match.get_input_output(matches_list_league=matches_list_league)
        if x == []:
            return None
        y_predict = self.predict(np.array([x]))[0]
        if y_predict[0] > 0.5: # L'ia prédit que le match ne sera pas BTS
            return True if y_true == [1, 0] else False
        else: # L'ia prédit que le match sera BTS
            return True if y_true == [0, 1] else False
        
nn = BTSPredictNeuralNetwork(42) #

list_league_to_test = ['PL20232024',
                       'LIGA20232024',
                       'SerieA20232024',
                       'BL20232024',
                       'L120232024']
x, y = data.get_all_datas(list_league_to_exclude=list_league_to_test)
print(x.shape)
#nn.load_weight()
nn.train(x, y)
score_total = 0
for league in list_league_to_test:
    score_total += nn.test_this_league(league)
    
print(f"Score total : {score_total:.2f}")
    