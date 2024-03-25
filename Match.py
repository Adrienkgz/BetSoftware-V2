from dataclasses import dataclass
from databasemanager import dbManager
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

@dataclass
class Match:
    """
    Data class for match data
    Stats for each match: Index, Date, HomeTeam, Score, AwayTeam, HomeBTS, HomeCS, HomeFTS, AwayBTS, AwayCS, AwayFTS, League (PL, Ligue1...), Season
    Possibly to test if the AI is not working well: %victory in the last 10 matches, for stats put the calculations on either all which corresponds to since the beginning of the season or a number of matches
    """
    index: int
    date: int # Number of seconds since 01/01/1970
    hour: str # in the format 2100 for 21:00
    home_team: str
    result: str # score in the format '1-1'
    away_team: str
    home_win_odd: float
    draw_odd: float
    away_win_odd: float
    bts_yes_odd: float
    bts_no_odd: float
    over_25_odd: float
    under_25_odd: float
    shot_home: int
    shot_away: int
    shot_on_goal_home: int
    shot_on_goal_away: int
    possession_home: int
    possession_away: int
    corner_home: int
    corner_away: int
    offside_home: int
    offside_away: int
    fouls_home: int
    fouls_away: int
    saves_home: int
    saves_away: int
    commentary: str
    weather: str
    temperature: int
    stadium: str
    id_league: str
    
    # Getter
    def get_winner(self):
            """
            Determines the winner of the match based on the result.

            Returns:
                int: 0 if the first team wins, 2 if the second team wins, 1 if it's a draw.
            """
            result_list = self.result.split('-')
            if int(result_list[0]) > int(result_list[1]):
                return self.home_team
            elif int(result_list[0]) < int(result_list[1]):
                return self.away_team
            else:
                return "Draw"
            
    def is_draw(self):
        """
        Determines if the match ended in a draw.

        Returns:
            int: 1 if the match ended in a draw, 0 otherwise.
        """
        result_list = self.result.split('-')
        return int(result_list[0]) == int(result_list[1])
    
    def get_bts(self):
        """
        Determines if both teams scored.

        Returns:
            int: 1 if both teams scored, 0 otherwise.
        """
        result_list = self.result.split('-')
        return (int(result_list[0]) > 0 and int(result_list[1])) > 0

    def get_over_25(self):
        """
        Determines if the match has over 2.5 goals.

        Returns:
            int: 1 if the match has over 2.5 goals, 0 otherwise.
        """
        result_list = self.result.split('-')
        return (int(result_list[0]) + int(result_list[1])) > 2
    
    def get_failed_to_score_home(self):
        """
        Determines if the home team failed to score.

        Returns:
            int: 1 if the home team failed to score, 0 otherwise.
        """
        result_list = self.result.split('-')
        failed_to_score = int(result_list[0]) == 0
        return 1 if failed_to_score else 0
    
    def get_failed_to_score_away(self):
        """
        Determines if the away team failed to score.

        Returns:
            int: 1 if the away team failed to score, 0 otherwise.
        """
        result_list = self.result.split('-')
        failed_to_score = int(result_list[1]) == 0
        return 1 if failed_to_score else 0
    
    def get_clean_sheet_home(self):
        """
        Determines if the home team kept a clean sheet.

        Returns:
            int: 1 if the home team kept a clean sheet, 0 otherwise.
        """
        result_list = self.result.split('-')
        clean_sheet = int(result_list[1]) == 0
        return 1 if clean_sheet else 0
    
    def get_clean_sheet_away(self):
        """
        Determines if the away team kept a clean sheet.

        Returns:
            int: 1 if the away team kept a clean sheet, 0 otherwise.
        """
        result_list = self.result.split('-')
        clean_sheet = int(result_list[0]) == 0
        return 1 if clean_sheet else 0
    
    def get_goals_scored_home(self):
        """
        Determines the number of goals scored by the home team.

        Returns:
            int: The number of goals scored by the home team.
        """
        result_list = self.result.split('-')
        return int(result_list[0])
    
    def get_goals_scored_away(self):
        """
        Determines the number of goals scored by the away team.

        Returns:
            int: The number of goals scored by the away team.
        """
        result_list = self.result.split('-')
        return int(result_list[1])
    
    def get_percentage_hour(self):
        """
        Determines the percentage of the hour of the match.

        Returns:
            int: The percentage of the hour of the match.
        """
        hour = self.hour.replace(":", "")
        return int(hour)/2400
    """ 
    Method to add the match to the database
    """
    
    def add_to_database(self):
            """
            Adds the match details to the database.

            Returns:
            None
            """
            if dbManager.apply_request_to_database(f"SELECT COUNT(*) FROM Matchs WHERE id_match = {self.index}")[0][0] == 0:
                
                query = "INSERT INTO Matchs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                values = (self.index, int(self.date), self.hour, self.home_team.replace("'", "''"), self.result, self.away_team.replace("'", "''"), self.home_win_odd, self.draw_odd, self.away_win_odd, self.bts_yes_odd, self.bts_no_odd, self.over_25_odd, self.under_25_odd, self.shot_home, self.shot_away, self.shot_on_goal_home, self.shot_on_goal_away, self.possession_home, self.possession_away, self.corner_home, self.corner_away, self.offside_home, self.offside_away, self.fouls_home, self.fouls_away, self.saves_home, self.saves_away, self.commentary.replace("''", "'"), self.weather, self.temperature, self.stadium, self.id_league)

                dbManager.cur.execute(query, values)
                dbManager.commit()

    """ 
    Method to prepare the input for the model
    """
    
    def get_n_last_matches(self, n:int = 10, list_matches_league:list = [], n_min:int = 10):
        """
        Retrieves the last n matches for the home team at home, away team at away,
        home team in general, and away team in general.

        Parameters:
        - n (int): Number of matches to retrieve (default is 10)

        Returns:
        - Tuple: A tuple containing the last home team matches at home,
             last away team matches at away, last home team matches in general,
             and last away team matches in general.
        """
        index = 0
        #On cherche l'id du match actuel
        for i, match in enumerate(list_matches_league):
            if match == self:
                list_matches_league = list_matches_league[:i] #On ne garde que les matchs précédents
                index = i
                break
            
        if len(list_matches_league) < n_min:
            return [], [], [], []
        
        # On récupère les n derniers matchs de l'équipe à domicile à domicile
        last_home_team_matches_at_home = [match for match in list_matches_league if match.home_team == self.home_team]
        if len(last_home_team_matches_at_home) > n:
            last_home_team_matches_at_home = last_home_team_matches_at_home[len(last_home_team_matches_at_home)-n:]
        
        if len(last_home_team_matches_at_home) < n_min:
            return [], [], [], []
        
        #ON récupere les n dderniers matchs de l'équipe à domicile en général
        last_home_team_matches_general = [match for match in list_matches_league if match.home_team == self.home_team or match.away_team == self.home_team]
        if len(last_home_team_matches_general) > n:
            last_home_team_matches_general = last_home_team_matches_general[len(last_home_team_matches_general)-n:]
            
        # On récupère les n derniers matchs de l'équipe à l'extérieur à l'extérieur
        last_away_team_matches_at_away = [match for match in list_matches_league if match.away_team == self.away_team]
        if len(last_away_team_matches_at_away) > n:
            last_away_team_matches_at_away = last_away_team_matches_at_away[len(last_away_team_matches_at_away)-n:]
        
        if len(last_away_team_matches_at_away) < n_min:
            return [], [], [], []
        # On récupère les n derniers matchs de l'équipe à l'extérieur en général
        last_away_team_matches_general = [match for match in list_matches_league if match.home_team == self.away_team or match.away_team == self.away_team]
        if len(last_away_team_matches_general) > n:
            last_away_team_matches_general = last_away_team_matches_general[len(last_away_team_matches_general)-n:]
        
        return last_home_team_matches_at_home, last_home_team_matches_general, last_away_team_matches_at_away, last_away_team_matches_general
    
    def get_X_input_intermediaire(self, team:str):
        """
        Returns the intermediate input features for a given team in a match.

        Parameters:
        - team (str): The name of the team.

        Returns:
        - X_inter (tuple): A tuple containing the following features:
            - 1 if the team won, 0 otherwise
            - 1 if the match ended in a draw, 0 otherwise
            - 1 if both teams scored, 0 otherwise
            - 1 if the team failed to score, 0 otherwise
            - 1 if the team kept a clean sheet, 0 otherwise
            - 1 if there were more than 2.5 goals, 0 otherwise
            - Number of goals scored by the team
            - Number of goals conceded by the team
            - Number of shots taken by the team
            - Number of shots taken against the team
            - Number of shots on target by the team
            - Number of shots on target against the team
            - Possession percentage of the team
            - Number of corners taken by the team
            - Number of corners conceded by the team
            - Number of offside calls against the team
            - Number of offside calls for the team
            - Number of fouls committed by the team
            - Number of fouls committed against the team
            - Number of saves made by the team
            - Number of saves made against the team
        """
        X_inter = [1 if self.get_winner() == team else 0, 
                   1 if self.is_draw() else 0, 
                   1 if self.get_bts() else 0, 
                   self.get_failed_to_score_home() if self.home_team == team else self.get_failed_to_score_away(), 
                   self.get_clean_sheet_home() if self.home_team == team else self.get_clean_sheet_away(), 
                   1 if self.get_over_25() else 0, 
                   self.get_goals_scored_home()/4 if self.home_team == team else self.get_goals_scored_away()/4, 
                   self.get_goals_scored_away()/4 if self.home_team == team else self.get_goals_scored_home()/4, 
                   self.shot_home/30 if self.home_team == team else self.shot_away/30, 
                   self.shot_away/30 if self.home_team == team else self.shot_home/30, 
                   self.shot_on_goal_home/20 if self.home_team == team else self.shot_on_goal_away/20, 
                   self.shot_on_goal_away/20 if self.home_team == team else self.shot_on_goal_home/20, 
                   self.possession_home/100 if self.home_team == team else self.possession_away/100, 
                   self.corner_home/15 if self.home_team == team else self.corner_away/15, 
                   self.corner_away/15 if self.home_team == team else self.corner_home/15, 
                   self.offside_home/5 if self.home_team == team else self.offside_away/5, 
                   self.offside_away/5 if self.home_team == team else self.offside_home/5, 
                   self.fouls_home/30 if self.home_team == team else self.fouls_away/30, 
                   self.fouls_away/30 if self.home_team == team else self.fouls_home/30, 
                   self.saves_home/7 if self.home_team == team else self.saves_away/7, 
                   self.saves_away/7 if self.home_team == team else self.saves_home/7]
        return X_inter
    
    def get_input_output(self, n:int = 10, matches_list_league:list = [], only_input:bool = False):
        
        # Get the last n matches for the home team at home, away team at away, home_team in general, and away team in general
        last_home_team_matches_at_home, last_home_team_matches_general, last_away_team_matches_at_away, last_away_team_matches_general = self.get_n_last_matches(n, matches_list_league)
        
        if last_home_team_matches_at_home == [] or last_home_team_matches_general == [] or last_away_team_matches_at_away == [] or last_away_team_matches_general == []:
            return [], []
        
        X_input = []
        X_inter = []
        
        # Add the input for the home team at home
        for match in last_home_team_matches_at_home:
            X_inter.append(match.get_X_input_intermediaire(self.home_team))
        X_input.append(X_inter)
        X_inter = []
        
        # Add the input for the away team at away
        for match in last_away_team_matches_at_away:
            X_inter.append(match.get_X_input_intermediaire(self.away_team))
        X_input.append(X_inter) 
        X_inter = []


        # Add the input for the home team in general
        for match in last_home_team_matches_general:
            X_inter.append(match.get_X_input_intermediaire(self.home_team))
        X_input.append(X_inter)
        X_inter = []

        # Add the input for the away team in general
        for match in last_away_team_matches_general:
            X_inter.append(match.get_X_input_intermediaire(self.away_team))
        X_input.append(X_inter)   
        
        # Add the temperature
        #X_input.append((self.temperature-15)/20)
        
        # Add the time of the match
        #X_input.append(self.get_percentage_hour())
        
        
        # Add the output
        
        if not only_input:
            output = [1 if self.get_bts() else 0]
        else:
            output = []
        return X_input, output
        
