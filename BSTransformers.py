
from keras.callbacks import EarlyStopping
import data 
import tensorflow as tf
import numpy as np
from tensorflow.nn import leaky_relu as LeakyRelu # type: ignore
from keras.layers import Dense

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
        input_shape_lstm = (5, 7, 24)  # (num_lists, num_matches, num_stats)
        number_parameter_dense = 3 # (num_stats,)
        
        # Define the LSTM layer
        lstm_layer =  tf.keras.layers.SimpleRNN(units=256, return_sequences=True)

        # Define the input layer
        input_layer = tf.keras.layers.Input(shape=input_shape_lstm)

        # Process each list of matches separately
        lstm_outputs = []
        for i in range(4):
            lstm_output = lstm_layer(input_layer[:, i, :, :])
            lstm_outputs.append(lstm_output)

        # Process the list of match inputs in parallel
        x_temp = tf.keras.layers.Dense(32, activation=LeakyRelu)(input_layer[:, 4, 0, :number_parameter_dense])
        
        
        
        # Concatenate the outputs
        concatenated_output = tf.keras.layers.Concatenate()(lstm_outputs)
        
        concatenated_lstm_and_dense = tf.keras.layers.Concatenate()([concatenated_output[:, 0, :], x_temp])
        
        
        x = Dense(32, activation=LeakyRelu)(concatenated_lstm_and_dense)
        x = Dense(16, activation=LeakyRelu)(x)
        
        
        
        # Define the output layer
        output_layer = Dense(2, activation='linear')(x)

        # Define the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        return model
    

    def train(self, x_training, y_training):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        self.model.fit(x=x_training, y=y_training, verbose=1, epochs=100, batch_size=32, validation_split=0.1, use_multiprocessing=False, callbacks=[early_stopping])
        
        
    def predict(self, x):
        
        return self.model.predict(x=x, verbose=0)
    
    def test_this_league(self, league_str, desired_output='bts', n:int=10):
        for league in data.get_all_league():
            if league.ID_League == league_str:
                league = league
                break
            
        list_matchs = league.get_list_matches()
        true_prediction, score, compteur_prediction = 0, 0, 0
        
        for match in list_matchs:
            have_good_prediction_bool = self.is_nn_have_good_predict(match, matches_list_league=list_matchs, desired_output=desired_output, n=n)
            if have_good_prediction_bool is None:
                continue
            compteur_prediction += 1
            score -= 1
            if have_good_prediction_bool:
                if desired_output == 'bts':
                    if match.get_bts() == 1:
                        true_prediction += 1
                        score += match.bts_yes_odd
                    else:
                        true_prediction += 1
                        score += match.bts_no_odd
                elif desired_output == '+2.5':
                    if match.get_over_25() == 1:
                        true_prediction += 1
                        score += match.over_25_odd
                    else:
                        true_prediction += 1
                        score += match.under_25_odd
                elif desired_output == 'winner':
                    if match.get_winner() == match.home_team:
                        true_prediction += 1
                        score += match.home_win_odd
                    elif match.get_winner() == "Draw":
                        true_prediction += 1
                        score += match.draw_odd
                    else:
                        true_prediction += 1
                        score += match.away_win_odd
        
        print(f"Test de la league {league.get_league_name()} pour la saison {league.get_season()} :")
        print(f"Pourcentage de prédiction juste : {true_prediction/compteur_prediction*100:.2f}%")
        print(f"Score : {score:.2f}")
        print(f"Nombre de matchs prédits : {compteur_prediction}\n")
        return score
    
    def is_nn_have_good_predict(self, match, desired_output:str,  matches_list_league=[], n=10):
        x, y_true = match.get_input_output(desired_output=desired_output, matches_list_league=matches_list_league, n=n)
        if x == []:
            return None
        y_predict = self.predict(np.array([x]))[0]
        if desired_output == 'bts':
            if y_predict[0] > 0.5: # L'ia prédit que le match ne sera pas BTS
                return True if y_true == [1, 0] else False
            else: # L'ia prédit que le match sera BTS
                return True if y_true == [0, 1] else False
        elif desired_output == '+2.5':
            if y_predict[0] > 0.5:
                return True if y_true == [1, 0] else False
            else:
                return True if y_true == [0, 1] else False
        elif desired_output == 'winner':
            if np.argmax(y_predict) == 0:
                return True if y_true == [1, 0, 0] else False
            elif np.argmax(y_predict) == 1:
                return True if y_true == [0, 1, 0] else False
            else:
                return True if y_true == [0, 0, 1] else False
        else:
            raise ValueError("desired_output must be 'bts' or '+2.5' or 'winner'")
        
nn = BTSPredictNeuralNetwork(42) #
n = 7
list_league_to_test = ['PL20232024',
                       'LIGA20232024',
                       'SerieA20232024',
                       'BL20232024',
                       'L120232024',]
desired_output = 'bts' # 'bts' or '+2.5' or 'winner'
x, y, _ = data.get_all_datas(list_league_to_exclude=list_league_to_test, desired_output=desired_output, n=n)
#nn.load_weight()
nn.train(x_training=x, y_training=y)
score_total = 0
for league in list_league_to_test:
    score_total += nn.test_this_league(league, desired_output=desired_output, n=n)
    
print(f"Score total : {score_total:.2f}")

    