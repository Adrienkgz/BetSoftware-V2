from databasemanager import dbManager
from Match import Match
from tqdm import tqdm
import time
from League import League
import numpy as np
import random

def get_all_league() -> list:
    """Get all the leagues in the database

    Returns:
        list: The list of all the leagues
    """
    league_list = []
    for league_tuple in  dbManager.apply_request_to_database("SELECT * FROM League;"):
        league_list.append(League(*league_tuple))
    return league_list

def get_balanced_data(x, y, matches_list, desired_output:str):
    if desired_output == 'bts' or desired_output == '+2.5':
        # On sépare les données en deux listes, une pour les BTS et une pour les non BTS
        yes_list, no_list, yes_matches_list, no_matches_list = [], [], [], []
        for i in range(len(x)):
            if y[i][1]  == 1: # Si le match est un BTS
                yes_list += [x[i]]
                yes_matches_list += [matches_list[i]]
            else:
                no_list += [x[i]]
                no_matches_list += [matches_list[i]]
        # On veut que les deux listes aient la même taille
        if len(yes_list) > len(no_list): # Si il y a plus de BTS que de non BTS
            random_indices = random.sample(range(len(yes_list)), len(no_list))
            yes_list = [yes_list[i] for i in random_indices]
            yes_matches_list = [yes_matches_list[i] for i in random_indices]
        elif len(no_list) > len(yes_list): # Si il y a plus de non BTS que de BTS
            random_indices = random.sample(range(len(no_list)), len(yes_list))
            no_list = [no_list[i] for i in random_indices]
            no_matches_list = [no_matches_list[i] for i in random_indices]
        # On regroupe les deux listes
        x = yes_list + no_list
        y = [[0, 1]]*len(yes_list) + [[1, 0]]*len(no_list)
        matches_list = yes_matches_list + no_matches_list
        
    elif desired_output == 'winner':
        # On sépare les données en trois listes, une pour les matchs gagnés par l'équipe à domicile, une pour les matchs nuls et une pour les matchs gagnés par l'équipe à l'extérieur
        home_list, draw_list, away_list, home_matches_list, draw_matches_list, away_matches_list = [], [], [], [], [], []
        for i in range(len(x)):
            if y[i][0]  == 1: # Si le match est un match gagné par l'équipe à domicile
                home_list += [x[i]]
                home_matches_list += [matches_list[i]]
            elif y[i][1] == 1: # Si le match est un match nul
                draw_list += [x[i]]
                draw_matches_list += [matches_list[i]]
            else:
                away_list += [x[i]]
                away_matches_list += [matches_list[i]]
        # On veut que les trois listes aient la même taille
        min_length = min(len(home_list), len(draw_list), len(away_list))
        home_list = home_list[:min_length]
        draw_list = draw_list[:min_length]
        away_list = away_list[:min_length]
        home_matches_list = home_matches_list[:min_length]
        draw_matches_list = draw_matches_list[:min_length]
        away_matches_list = away_matches_list[:min_length]
        # On regroupe les trois listes
        x = home_list + draw_list + away_list
        y = [[1, 0, 0]]*len(home_list) + [[0, 1, 0]]*len(draw_list) + [[0, 0, 1]]*len(away_list)
        matches_list = home_matches_list + draw_matches_list + away_matches_list
    else:
        raise ValueError("The desired output must be 'bts', '+2.5' or 'winner'")
    
    return x, y, matches_list
            

def get_all_datas(show_progress:bool=True, list_league_to_exclude:list[str] = [], desired_output:str = 'bts', only_big_five_league:bool = False, only_this_league:str = '', only_this_specific_league:str = '', only_this_leagues:list=[], balanced_data:bool = False, n:int=7) -> tuple:
    x_total, y_total, matches_list_total = [], [], []
    list_leagues = get_all_league()
    progress_bar = tqdm(list_leagues) if show_progress else list_leagues
    for league in progress_bar:
        if only_big_five_league and league.get_league_name() not in ['PL', 'LIGA', 'SerieA', 'BL', 'L1']:
            continue
        elif only_this_league != '' and league.get_league_name() != only_this_league:
            continue
        elif only_this_specific_league != '' and league.ID_League != only_this_specific_league:
            continue
        elif only_this_leagues != [] and league.ID_League not in only_this_leagues:
            continue
        elif league.ID_League in list_league_to_exclude:
            continue
        progress_bar.set_description(f"Récupération des données pour la ligue {league.ID_League}")
        x, y, match = league.get_input_output_datas(show_progress=False, desired_output=desired_output, n=n) # On récupère les données de la ligue
        if balanced_data:
            x, y, match = get_balanced_data(x, y, match, desired_output)
        x_total += x
        y_total += y
        matches_list_total += match
    
    x_total = np.array(x_total)
    y_total = np.array(y_total)
    return x_total, y_total, matches_list_total

def get_con
