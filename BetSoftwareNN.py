from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
import data 
import tensorflow as tf
import numpy as np
import os

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

        # Initialiser le modèle séquentiel
        model = tf.keras.Sequential()

        # Ajouter la couche d'entrée (Remplacez shape avec votre forme d'entrée réelle)
        model.add(tf.keras.layers.InputLayer(input_shape=(4, 10, 21)))

        # Ajouter une couche de convolution avec 32 filtres, une taille de noyau de 3x3 et une activation ReLU
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))

        # Ajouter une couche de pooling pour réduire la dimensionnalité
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        
        # Couche LSTM bidirectionnelle pour capturer les dépendances temporelles
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128)))

        # Couche dense pour la classification
        for _ in range(7):
            model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
            model.add(tf.keras.layers.Dropout(0.15))

        # Couche de sortie avec activation sigmoid pour la classification binaire
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))

        # Compiler le modèle avec la fonction de perte binary_crossentropy
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Afficher les valeurs de x à chaque étape
        model.summary()

        return model

    def train(self, x_training, y_training):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(x_training, y_training, verbose=0, epochs=10, batch_size=256, validation_split=0.2, use_multiprocessing=False, callbacks=[early_stopping])
        
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
            prediction = self.is_this_match_will_have_bts(match, matches_list_league=list_matchs)
            if prediction is None:
                continue
            compteur_prediction += 1
            score -= 1
            if prediction:
                if match.get_bts():
                    true_prediction += 1
                    score += match.bts_yes_odd
            else:
                if not match.get_bts():
                    true_prediction += 1
                    score += match.bts_no_odd
        
        print(f"Test de la league {league.get_league_name()} pour la saison {league.get_season()} :")
        print(f"Pourcentage de prédiction juste : {true_prediction/compteur_prediction*100}%")
        print(f"Score : {score}")
    
    def is_this_match_will_have_bts(self, match, matches_list_league=[]):
        x, y_true = match.get_input_output(matches_list_league=matches_list_league)
        if x == []:
            return None
        y_predict = self.predict(np.array([x]))
        if y_predict > 0.5:
            return True
        else:
            return False
        
nn = BTSPredictNeuralNetwork(42) #

list_league_to_test = ['BL20222023', 
                       'PL20222023', 
                       'LIGA20222023', 
                       'SerieA20222023', 
                       'L120222023', 
                       'EREDIVISIE20222023', 
                       'EREDIVISIE20212022', 
                       'EREDIVISIE20202021', 
                       'PL20232024',
                       'LIGA20232024',
                       'SerieA20232024',
                       'BL20232024',
                       'L120232024']
x, y = data.get_all_datas(list_league_to_exclude=list_league_to_test)
#nn.load_weight()
nn.model.fit(x=x, y=y, epochs=10, batch_size=256, validation_split=0.2, use_multiprocessing=True, verbose=1)
for league in list_league_to_test:
    nn.test_this_league(league)
    