import gym
from gym import spaces
import numpy as np
import data

class BetEnv(gym.Env):
    def __init__(self, desired_output, training_mode = True, list_leagues_to_exclude=[], payroll=200, only_league_params:str = None, praising_coeff=1.0):
        super(BetEnv, self).__init__()
        #possible actions : bet on home team, bet on draw, bet on away team, do not bet, for bts and +2.5 bet on yes, bet on no, do not bet
        self.action_space = spaces.Discrete(4) if desired_output == 'winner' else spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 10, 21), dtype=np.float32)
        self.reward_range = (-10, 10)
        self.list_leagues_to_exclude = list_leagues_to_exclude
        self.desired_output = desired_output
        self.data_x, self.data_y, self.data_matches = self.init_db(only_league_params)
        self.current_match, self.current_output, self.current_index = None, None, None
        self.training_mode = training_mode
        self.payroll = payroll
        self.praising_coeff = praising_coeff
        self.reset()
    
    def init_db(self, only_league_param = None):
            """
            Initializes the database by retrieving all data from the specified leagues.

            Args:
                self (Env): The current instance of the Env class.

            Returns:
                list: A list of all data retrieved from the specified leagues.
            """
            if only_league_param is not None:
                return data.get_all_datas(only_this_specific_league=only_league_param, desired_output=self.desired_output)
            return data.get_all_datas(list_league_to_exclude=self.list_leagues_to_exclude, desired_output=self.desired_output)
    
    def get_a_random_match(self):
        """
        Returns a random match from the dataset.

        Returns:
            numpy.ndarray: The input data for the selected match.
        """
        index = np.random.randint(0, self.data_x.shape[0])
        self.current_match = self.data_matches[index]
        self.current_output = self.data_y[index]
        input_to_return = self.data_x[index]
        input_to_return[4][0][3] = self.payroll
        if self.desired_output == 'winner':
            input_to_return[4][0][4] = self.current_match.home_win_odd
            input_to_return[4][0][5] = self.current_match.draw_odd
            input_to_return[4][0][6] = self.current_match.away_win_odd
        else:
            input_to_return[4][0][4] = self.current_match.bts_yes_odd if self.desired_output == 'bts' else self.current_match.over_25_odd
            input_to_return[4][0][5] = self.current_match.bts_no_odd if self.desired_output == 'bts' else self.current_match.under_25_odd
        return input_to_return
    
    def agent_bet(self):
        """
        Decreases the payroll by 1 unit.

        This method is used to deduct 1 unit from the payroll attribute of the class instance.

        Parameters:
            self (YourClassName): The instance of the class.

        Returns:
            None
        """
        self.payroll -= 1
        
    
    def agent_win(self, odd):
        """
        Updates the payroll by adding the given odd.

        Args:
            odd (float): The odd to be added to the payroll.

        Returns:
            None
        """
        self.payroll += odd
        
    def reset(self):
        if self.training_mode:
            return self.get_a_random_match()
        self.current_index = 0
        self.current_match = self.data_matches[self.current_index]
        self.current_output = self.data_y[self.current_index]
        input_to_return = self.data_x[self.current_index]
        input_to_return[4][0][3] = self.payroll
        if self.desired_output == 'winner':
            input_to_return[4][0][4] = self.current_match.home_win_odd
            input_to_return[4][0][5] = self.current_match.draw_odd
            input_to_return[4][0][6] = self.current_match.away_win_odd
        else:
            input_to_return[4][0][4] = self.current_match.bts_yes_odd if self.desired_output == 'bts' else self.current_match.over_25_odd
            input_to_return[4][0][5] = self.current_match.bts_no_odd if self.desired_output == 'bts' else self.current_match.under_25_odd
        return input_to_return
    
    def go_on_next_match(self):
        self.current_index += 1
        if self.current_index >= len(self.data_x):
            return None
        self.current_match = self.data_matches[self.current_index]
        self.current_output = self.data_y[self.current_index]
        input_to_return = self.data_x[self.current_index]
        input_to_return[4][0][3] = self.payroll
        if self.desired_output == 'winner':
            input_to_return[4][0][4] = self.current_match.home_win_odd
            input_to_return[4][0][5] = self.current_match.draw_odd
            input_to_return[4][0][6] = self.current_match.away_win_odd
        else:
            input_to_return[4][0][4] = self.current_match.bts_yes_odd if self.desired_output == 'bts' else self.current_match.over_25_odd
            input_to_return[4][0][5] = self.current_match.bts_no_odd if self.desired_output == 'bts' else self.current_match.under_25_odd
        return input_to_return
    
    def step(self, action):
        """
        Executes a step in the environment based on the given action. If the desired output of the env is 'winner', the action space is 4, otherwise it is 3.
        For the 'winner' output, the actions are as follows:
            0: Do nothing.
            1: Bet on home team.
            2: Bet on draw.
            3: Bet on away team.
        For the 'bts' and '+2.5' outputs, the actions are as follows:
            0: Do nothing.
            1: Bet on yes.
            2: Bet on no.

        Parameters:
            action (int): The action to take in the environment.

        Returns:
            tuple: A tuple containing the following elements:
                - observation (object): The new observation after taking the action.
                - reward (float): The reward obtained from the action.
                - done (bool): A flag indicating if the episode is done.
                - info (dict): Additional information about the step.
        """
        reward = 0
        if (self.desired_output == 'winner'):
            if action == 0: # do nothing
                return self.reset(), 0, False, {}
            else:
                # Agent has made a bet
                self.agent_bet()
                if action == 1: # bet on home team
                    if self.current_output[0] == 1:
                        self.agent_win(self.current_match.home_win_odd)
                        reward = self.current_match.home_win_odd - 1
                    else:
                        reward = -1
                        return self.reset(), -1, False, {}
                elif action == 2: # bet on draw
                    if self.current_output[1] == 1:
                        self.agent_win(self.current_match.draw_odd)
                        reward = self.current_match.draw_odd - 1
                    else:
                        reward = -1
                elif action == 3: # bet on away team
                    if self.current_output[2] == 1:
                        self.agent_win(self.current_match.away_win_odd)
                        reward = self.current_match.away_win_odd - 1
                    else:
                        reward = -1
        else:
            if action == 0: # do nothing
                reward = 0
            else:
                self.agent_bet()
                if action == 1: # bet on yes
                    if self.current_output[1] == 1:
                        reward = self.current_match.bts_yes_odd - 1 if self.desired_output == 'bts' else self.current_match.over_25_odd - 1
                        self.agent_win(reward+1)
                    else:
                        reward = -1
                elif action == 2: # bet on no
                    if self.current_output[0] == 1:
                        reward = self.current_match.bts_no_odd - 1 if self.desired_output == 'bts' else self.current_match.under_25_odd - 1
                        self.agent_win(reward+1)
                    else:
                        reward = -1
        reward *= self.praising_coeff if reward > 0 else 1
        if self.training_mode:
            return self.reset(), reward, False, {}
        else:
            next_input = self.go_on_next_match()
            if next_input is None or self.payroll <= 0:
                return next_input, reward, True, {}
            return next_input, reward, False, {}

        
        