def test_this_league(league, model):
    x_inputs, y_true, list_matchs = league.get_input_output_datas()
    y_pred = []
    for input in x_inputs:
        prediction = nn.predict(input)
        y_pred.append(prediction)
    
    true_prediction = 0
    score = 0
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5: #Si l'ia a prédit un bts sur ce match
            if list_matchs[i].isBothTeamsScored():
                true_prediction += 1
                score += list_matchs[i].bts_yes_odd
            else:
                score -= 1
        else:
            if not list_matchs[i].isBothTeamsScored():
                true_prediction += 1
                score += list_matchs[i].bts_no_odd
            else:
                score -= 1
                
    print(f"Test de la league {league.get_league_name()} pour la saison {league.get_season()} :")
    print(f"Pourcentage de prédiction juste : {true_prediction/len(y_pred)*100}%")
    print(f"Score : {score}")
            
        