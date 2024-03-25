from databasemanager import dbManager
from Match import Match
from tqdm import tqdm
import time
from League import League
import numpy as np

def get_all_league() -> list:
    """Get all the leagues in the database

    Returns:
        list: The list of all the leagues
    """
    league_list = []
    for league_tuple in  dbManager.apply_request_to_database("SELECT * FROM League;"):
        league_list.append(League(*league_tuple))
    return league_list


def get_all_datas(show_progress:bool=True, list_league_to_exclude:list[str] = []) -> tuple:
    x_total, y_total = [], []
    list_leagues = get_all_league()
    progress_bar = tqdm(list_leagues) if show_progress else list_leagues
    for league in progress_bar:
        if league.ID_League in list_league_to_exclude:
            continue
        progress_bar.set_description(f"Récupération des données pour la ligue {league.ID_League}")
        x, y, _ = league.get_input_output_datas(show_progress=False)
        x_total += x
        y_total += y
    x_total = np.array(x_total)
    y_total = np.array(y_total)
    return x_total, y_total


    