from dataclasses import dataclass

from databasemanager import dbManager

@dataclass
class Player:
    """
    Represents a player in a match.
    
    Attributes:
        id_player (int): The ID of the player.
        name (str): The last name of the player.
        first_name (str): The first name of the player.
        id_match (int): The ID of the match the player participated in.
        sub_in (int): The minute the player was substituted in.
        sub_out (int): The minute the player was substituted out.
        starter (bool): Indicates whether the player started the match.
        football_position (str): The position of the player in football.
        total_shot (int): The total number of shots taken by the player.
        shot_og (int): The number of shots on goal by the player.
        goals (int): The number of goals scored by the player.
        assists (int): The number of assists made by the player.
        fouls_commited (int): The number of fouls committed by the player.
        fouls_drawn (int): The number of fouls drawn by the player.
        offsides (int): The number of offsides by the player.
        penalty_miss (int): The number of penalties missed by the player.
        penalty_score (int): The number of penalties scored by the player.
        yellow_cards (int): The number of yellow cards received by the player.
        red_cards (int): The number of red cards received by the player.
        saves (int): The number of saves made by the player.
    """ 
    id_player:int
    name:str
    first_name:str
    id_match:int
    sub_in:int
    sub_out:int
    starter:bool
    football_position:str
    total_shot:int
    shot_og:int
    goals:int
    assists:int
    fouls_commited:int
    fouls_drawn:int
    offsides:int
    penalty_miss:int
    penalty_score:int
    yellow_cards:int
    red_cards:int
    saves:int
    team:str
    
    def __post_init__(self):
        self.name = self.name.replace("'", "")
        self.first_name = self.first_name.replace("'", "")
    def get_minutes_played(self):
        """
        Calculates the number of minutes played by the player in the match.
        
        Returns:
            int: The number of minutes played.
        """
        return self.sub_out - self.sub_in
    
    def add_to_database(self, match):
        """
        Adds the player to the database if they are not already present.
        
        Args:
            match: The match object the player participated in.
        """
        
        # Add the player into Players table if they are not already present
        try:
            if dbManager.apply_request_to_database(f"SELECT COUNT(*) FROM Players WHERE id_player = {self.id_player}")[0][0] == 0:
                dbManager.apply_request_to_database(f"INSERT INTO Players VALUES ({self.id_player}, '{self.name}', '{self.first_name}')")
            
            # Add the player into the play_in table if they are not already present
            if dbManager.apply_request_to_database(f"SELECT COUNT(*) FROM play_in WHERE id_player = {self.id_player} AND id_match = {match.index}")[0][0] == 0:
                self.team = self.team.replace("'", "")
                dbManager.apply_request_to_database(f"INSERT INTO play_in VALUES ({self.id_match}, {self.id_player}, {self.sub_in}, {self.sub_out}, {int(self.starter)}, '{self.football_position}', {self.total_shot}, {self.shot_og}, {self.goals}, {self.assists}, {self.fouls_commited}, {self.fouls_drawn}, {self.offsides}, {self.penalty_miss}, {self.penalty_score}, {self.yellow_cards}, {self.red_cards}, {self.saves}, '{self.team}')")
            
            dbManager.commit()
        except Exception as e:
            # In case of a problem adding a player, we delete the match from the database
            dbManager.apply_request_to_database(f"DELETE FROM Matchs WHERE id_match = {match.index}")
            dbManager.commit()
            raise e
            
         