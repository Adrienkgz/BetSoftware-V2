
from keras.callbacks import EarlyStopping
import data 
import tensorflow as tf
import numpy as np
from tensorflow.nn import leaky_relu as LeakyRelu # type: ignore
from keras.layers import Dense
import torch
import data
import numpy as np
from torch import nn
import torch.optim as optim

FILE_PATH_WEIGHT_BTSNN = "WeightBetsoftwareAIV1.pt"

class BTSPredictNeuralNetwork(nn.Module):
    def __init__(self, number_parameter):
        super(BTSPredictNeuralNetwork, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=24, hidden_size=256, num_layers=1, batch_first=True)
        self.x = nn.Sequential(
            nn.Linear(384, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
        )
        self.output_layer = nn.Linear(16, 2)
    
    def save_weight(self):
        """Save the weights of the neural network
        """
        print("Sauvegarde des poids du modèle en cours, ne fermez pas !")
        torch.save(self.state_dict(), FILE_PATH_WEIGHT_BTSNN)
        print("Sauvegarde effectué")
    
    def load_weight(self):
        self.load_state_dict(torch.load(FILE_PATH_WEIGHT_BTSNN))
        
    def forward(self, x):
        lstm_outputs = []
        for i in range(x.size(1)):
            lstm_output, _ = self.lstm_layer(x[:, i, :, :].float())
            lstm_outputs.append(lstm_output)
    
        x_temp = nn.functional.leaky_relu(x[:, 4, 0, :])
        concatenated_output = torch.cat(lstm_outputs, dim=1)
        concatenated_lstm_and_dense = torch.cat([concatenated_output[:, -1, :], x_temp], dim=1)
        x = self.x(concatenated_lstm_and_dense.float())
        output_layer = self.output_layer(x)
        return output_layer
    
    def train(self, x_training, y_training):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        num_epochs = 100
        batch_size = 32
        for epoch in range(num_epochs):
            for i in range(0, len(x_training), batch_size):
                inputs = torch.tensor(x_training[i:i+batch_size])
                labels = torch.tensor(y_training[i:i+batch_size])
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
    def predict(self, x):
        return self(x)
    
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
        y_predict = self.predict(torch.tensor([x]))[0]
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
        
btsnn = BTSPredictNeuralNetwork(42) #
n = 7
list_league_to_test = ['PL20232024',
                       'LIGA20232024',
                       'SerieA20232024',
                       'BL20232024',
                       'L120232024',]
desired_output = 'bts' # 'bts' or '+2.5' or 'winner'
x, y, _ = data.get_all_datas(list_league_to_exclude=list_league_to_test, desired_output=desired_output, n=n)
#nn.load_weight()
btsnn.train(x_training=x, y_training=y)
score_total = 0
for league in list_league_to_test:
    score_total += btsnn.test_this_league(league, desired_output=desired_output, n=n)
    
print(f"Score total : {score_total:.2f}")

    