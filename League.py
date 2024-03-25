from dataclasses import dataclass
from databasemanager import dbManager
from Match import Match
from tqdm import tqdm

@dataclass
class League:
    ID_League:str
    isdone:bool
    forebet_url:str = ''
    time_zone_difference:int = 0
    update_database = True
        
    def get_season(self):
        return self.ID_League[-8:]
    
    def get_league_name(self):
        return self.ID_League[:-8]
    
    def add_to_database(self):
        if dbManager.apply_request_to_database(f"SELECT COUNT(*) FROM League WHERE id_league = '{self.ID_League}'")[0][0] == 0:
            dbManager.apply_request_to_database(f"INSERT INTO League VALUES ('{self.ID_League}', '{self.forebet_url}', {self.isdone}, {self.time_zone_difference});")
            dbManager.commit()
        
    def get_list_matches(self):
        matches_tuple = dbManager.apply_request_to_database(f"SELECT * FROM Matchs WHERE id_league = '{self.ID_League}' ORDER BY matchday ASC;")
        return [Match(*match) for match in matches_tuple]
    
    def get_input_output_datas(self, n:int=10, show_progress:bool = False) -> list:
        """Get the datas input for this league

        Args:
            n (int, optional): The number of matches to get. Defaults to 10.

        Returns:
            list: The datas input for this league
        """
        
        matches_list = self.get_list_matches()
        matches_list_with_data = []
        #On récupère les données
        input_datas_set, output_datas_set = [], []
        progress_bar = tqdm(matches_list) if show_progress else matches_list
        for match in progress_bar:
            if match.home_team == "Sevilla FC" and match.away_team == "Deportivo Alavés":
                print()
            match_input, match_output = match.get_input_output(n, matches_list)
            if match_input == []:
                continue
            
            input_datas_set.append(match_input)
            output_datas_set.append(match_output)
            matches_list_with_data.append(match)
            
        return input_datas_set, output_datas_set, matches_list_with_data


            
