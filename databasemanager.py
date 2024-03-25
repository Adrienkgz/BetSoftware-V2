import sqlite3
#from League import League
import sqlite3

class DbManager:
    def __init__(self) -> None:
        self.init_cursor()
      
    def init_cursor(self):
        """
        Initializes the database connection and cursor.

        This method connects to the 'BetSoftware.sql' database file and creates a cursor object
        for executing SQL queries.

        Args:
            None

        Returns:
            None
        """
        self.conn = sqlite3.connect('BetSoftware.sql')
        self.cur = self.conn.cursor()
        
    def close_cursor(self):
        """
        Closes the database cursor and commits any pending changes.

        This method should be called when you are done executing queries and
        performing operations on the database. It ensures that any pending
        changes are committed to the database and releases the cursor resources.

        Note: After calling this method, you won't be able to execute any more
        queries using the same cursor.

        Returns:
            None
        """
        self.conn.commit()
        self.cur.close()
        
    def apply_request_to_database(self, request:str):
        """Apply request to the database and return the fetchall

        Args:
            request (str): The SQL request to be executed

        Returns:
            tuple: The result of the SQL request
        """
        self.cur.execute(request)
        result = self.cur.fetchall()
        return result
        
    def commit(self):
        self.conn.commit()

    def get_league_list_in_the_database(self):
        """Get the list of leagues in the database

        Returns:
            list: The list of league IDs
        """
        return self.apply_request_to_database("SELECT ID_league FROM LEAGUE")
    
    def remove_league_matchs(self, league):
        """Remove all matches and the league from the database

        Args:
            league: The league object to be removed
        """
        # Delete all the matches from the league in the Matchs table
        self.apply_request_to_database(f"DELETE FROM Matchs WHERE ID_league = '{league.ID_League}';")
        # Delete the league from the League table
        self.apply_request_to_database(f"DELETE FROM League WHERE ID_league = '{league.ID_League}';")
        
        # Delete all the teams from the league (missing code)

    def is_this_match_on_the_database(self, id_match):
        """Check if the match is already in the database

        Args:
            match: The match object to be checked

        Returns:
            bool: True if the match is in the database, False otherwise
        """
        return self.apply_request_to_database(f"SELECT COUNT(*) FROM Matchs WHERE ID_match = {id_match}")[0][0] == 1
    
  
            
dbManager = DbManager()

