"""

    This code is used for web scraping to obtain match information
    
"""
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from Player import Player
from Match import Match
from databasemanager import dbManager
import requests
from requests_html import HTMLSession
from League import League
import asyncio
import time
import aiohttp

headers = {
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
session = HTMLSession()      
        
        
async def get_html_code(url:str, relative_url:bool = False):
    """
    Fetches the HTML code from the specified URL.

    Args:
        url (str): The URL to fetch the HTML code from.

    Returns:
        str: The HTML code of the webpage, or an empty string if there was an error.
    """
    if relative_url:
        url = 'https://www.forebet.com' + url
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    return ""
    except Exception as e:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        return ""
        except Exception as e:
            print(f"Error fetching the URL: {e} {url}")
            return ""
                        
async def get_all_matchs_league_data_brut(league: League, refresh_base_data_bool: bool = True, debug: bool = False):
    """
    Retrieves all matches from a league using a parent link and updates the database.

    Args:
        league (League): The league object representing the league to retrieve matches from.
        refresh_base_data_bool (bool, optional): Indicates whether the data should be refreshed in the database. Default is True.
        debug (bool, optional): Indicates whether debug mode is enabled. Default is False.
    """
    # Adding the league to the database
    league.add_to_database()
    
    # Retrieving the HTML code of the page
    html_code = await get_html_code(league.forebet_url)
    
    # Analyzing the HTML code using BeautifulSoup
    soup = BeautifulSoup(html_code, 'lxml')
    
    # Initializing a list to store utility elements
    utility_elements = []
    
    # Retrieving the utility elements
    for element in soup.select('.haodd span'):
        try:
            if element.text != '':
                utility_elements.append(float(element.text))
        except:
            continue
            
    # Retrieving the navigation for the next pages
    nav = await get_nav(html_code=html_code)
    
    # Displaying a message to indicate the initialization of the progress bar
    print(f"Initializing the progress bar for the league: {league.ID_League}")
    
    # Retrieving the list of matches from the result page
    matchs_list = await get_data_from_the_result_page(html_code=await get_html_code(league.forebet_url))
    
    # If debug mode is not enabled, also retrieve matches from the next pages
    if not debug:
        for link in nav:
            matchs_list += await get_data_from_the_result_page(html_code=await get_html_code(link, relative_url=True))
    
    # Initializing the progress bar
    # Create a list of tasks for each match
    tasks = [extract_all_the_infos_for_this_match(match_infos, league.ID_League) for match_infos in matchs_list]

    # Run the tasks asynchronously
    await asyncio.gather(*tasks)
            
        
async def get_nav(html_code:str):
    """
    Retrieves the links of all web pages containing matches for the league and season specified in the `html_code`.

    Args:
        html_code (str): The HTML code of the parent page of the league.

    Returns:
        list: A list containing the links of the web pages.
    """
    nav = []
    # Initialize BeautifulSoup
    soup = BeautifulSoup(html_code, 'lxml')

    # Find clickable elements that are part of the navigation
    a_elements = soup.find_all('a', class_='pagenav')
        
    # Iterate through the elements, check if the links are not already in `nav`, and extract their links if not
    for a in a_elements:
        # Extract the link and add it to `nav` if it's not already present
        link = a['href']
        if link not in nav:
            nav.append(link)
        if a['href'] not in nav:
            nav.append(a['href'])

    return nav
                
async def get_data_from_the_result_page(html_code:str):
      
    # Initialize BeautifulSoup
    soup = BeautifulSoup(html_code, 'lxml')
            
    elements_list = soup.select('.heading, .tr_0, .tr_1')
                     
    # Iterate through the rows, create data classes, and add them to the data list
    date = ''
    list_matchs = []
    for row in elements_list:
        if len(row["class"]) == 1:
            if row["class"][0] == 'heading':
                day, month, year = row.text.split('.')
                date = year + month + day
            else:
                hour, home_team, result, away_team = await unpack_row(row.text)
                url_forebet = row.select('a')[1]["href"]
                list_matchs.append((date, hour, home_team, result, away_team, url_forebet))
                
    return list_matchs
        
async def extract_all_the_infos_for_this_match(infos_match:tuple, id_league):
    if dbManager.is_this_match_on_the_database(int(infos_match[5].rsplit('-', 1)[-1])):
        return
    home_odd, draw_odd, away_odd, under25, over25, bts_yes, bts_no, weather, temperature = await get_prediction_page_data_for_the_match(infos_match[5])
    print(f"Match {infos_match[2]} vs {infos_match[4]} : {home_odd} {draw_odd} {away_odd} {under25} {over25} {bts_yes} {bts_no} {weather} {temperature}")
    list_of_the_player, stadium, commentary, shot_home, shot_away, shot_og_home, shot_og_away, possession_home, possession_away, corner_home, corner_away, offside_home, offside_away, fouls_home, fouls_away, saves_home, saves_away = await get_match_center_data_from_this_match(infos_match[5])
    print(f"Match {infos_match[2]} vs {infos_match[4]} : {shot_home} {shot_away} {shot_og_home} {shot_og_away} {possession_home} {possession_away} {corner_home} {corner_away} {offside_home} {offside_away} {fouls_home} {fouls_away} {saves_home} {saves_away}")
    id_match = int(infos_match[5].rsplit('-', 1)[-1])
    match = Match(index=id_match,
                    date=infos_match[0], 
                    hour=infos_match[1], 
                    home_team=infos_match[2], 
                    result=infos_match[3], 
                    away_team=infos_match[4], 
                    home_win_odd=home_odd, 
                    draw_odd=draw_odd, 
                    away_win_odd=away_odd, 
                    bts_yes_odd=bts_yes, 
                    bts_no_odd=bts_no, 
                    over_25_odd=over25, 
                    under_25_odd=under25,
                    shot_home=shot_home,
                    shot_away=shot_away,
                    shot_on_goal_home=shot_og_home,
                    shot_on_goal_away=shot_og_away,
                    possession_home=possession_home,
                    possession_away=possession_away,
                    corner_home=corner_home,
                    corner_away=corner_away,
                    offside_home=offside_home,
                    offside_away=offside_away,
                    fouls_home=fouls_home,
                    fouls_away=fouls_away,
                    saves_home=saves_home,
                    saves_away=saves_away,
                    commentary=commentary,
                    weather=weather,
                    temperature=temperature,
                    stadium=stadium,
                    id_league=id_league)
    match.add_to_database()
    print(f"Match added to the database : {infos_match[2]} vs {infos_match[4]}")
    for player in list_of_the_player:
        player.add_to_database(match)
    print(f"Players added to the database for the match {infos_match[2]} vs {infos_match[4]}")
                    
async def get_heading_indices(rows_datas: list) -> list[int]:
    """Returns a list of indices of the headings that contain the dates of the matches.

    Args:
        rows_datas (list): List of row data from the HTML page.

    Returns:
        list[int]: List of indices of the headings.
    """
    heading_indices = []
    for i, row in enumerate(rows_datas):
        # Check if the element has the class 'heading'
        if row.get_attribute("class") == 'heading':
            # If so, add the index to the list
            heading_indices.append(i)
    return heading_indices
    
async def unpack_row(row_data_text: str):
    """Extracts relevant information from the raw data in the rows and returns them as variables.

    Args:
        row_data_text (str): Raw data from the row.

    Returns:
        hour (int): Match hour in the format 2100 for 21:00, for example.
        home_team (str): Home team name.
        result (str): Match result in the format '1-1'.
        away_team (str): Away team name.
    """
    row_datas_list = row_data_text.split()

    # Hour
    hour = row_datas_list[0]  # Add the hour
    del row_datas_list[0]  # Remove the hour from row_datas_list

    # Home Team
    while row_datas_list[1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        row_datas_list[0] = row_datas_list[0] + ' ' + row_datas_list[1]
        del row_datas_list[1]
    home_team = row_datas_list[0].replace("'", "")
    del row_datas_list[0]

    # Result
    while row_datas_list[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
        row_datas_list[0] = row_datas_list[0] + ' ' + row_datas_list[1]
        del row_datas_list[1]
    result = row_datas_list[0]
    del row_datas_list[0]

    # Away Team
    while len(row_datas_list) > 1:
        row_datas_list[0] = row_datas_list[0] + ' ' + row_datas_list[1]
        del row_datas_list[1]
    away_team =  row_datas_list[0].replace("'", "")

    return hour, home_team, result, away_team

async def get_prediction_page_data_for_the_match(url:str):
    """
    Retrieves prediction page data for a specific match.

    Args:
        url (str): The URL of the prediction page for the match.

    Returns:
        list: A list containing the following elements:
            - Home team odds
            - Draw odds
            - Away team odds
            - Under 2.5 goals odds
            - Over 2.5 goals odds
            - Both teams to score (Yes) odds
            - Both teams to score (No) odds
            - Weather condition
            - Temperature
    """
    url = "https://www.forebet.com/" + url
    html_code = await get_html_code(url)
    soup = BeautifulSoup(html_code, 'lxml')
    util_elements = []
    elements = soup.select('.haodd')
    try:
        hda_odds = elements[0] # Home Draw Away odds
        under_or_over_odds = elements[2]
        bts_odds = elements[3]
    
        for odd in hda_odds:
            try:
                if odd.text != '' and float(odd.text) > 1:
                    util_elements.append(float(odd.text))
            except:
                continue
        if len(util_elements) != 3:
            util_elements = [-1, -1, -1]
        util_elements_intermediate = []
        for odd in under_or_over_odds:
            try:
                if odd.text != '' and float(odd.text) > 1:
                    util_elements_intermediate.append(float(odd.text))
            except:
                continue
        if len(util_elements_intermediate) != 2:
            util_elements = util_elements + [-1, -1]
        else:
            util_elements = util_elements + util_elements_intermediate
                
        util_elements_intermediate = []
        for odd in bts_odds:
            try:
                if odd.text != '' and float(odd.text) > 1:
                    util_elements_intermediate.append(float(odd.text))
            except:
                continue
        if len(util_elements_intermediate) != 2:
            util_elements = util_elements + [-1, -1]
        else:
            util_elements = util_elements + util_elements_intermediate
                
        # Home, draw, away, under 2.5, over 2.5, both teams to score (Yes), both teams to score (No)
    except:
        util_elements = [-1, -1, -1, -1, -1, -1, -1]
    
    # Get weather and temperature
    try:
        weather_div = soup.select_one(".weather_main_pr")
        # Get weather
        util_elements.append(weather_div.select_one(".wthc")['src'][8:-4])
        
        # Get temperature
        text_temperature = weather_div.select_one('span').text.replace('°', '').replace(' ', '')
        done = False
        while not done:
            try:
                util_elements.append(int(text_temperature[-1]))
                done = True
            except:
                text_temperature = text_temperature[:-1]
        

    except:
        util_elements = util_elements + ["", -1]
    return util_elements
    

async def get_match_center_data_from_this_match(url:str):
    """
    Extracts essential match data from the given URL.

    Args:
        url (str): The URL of the match to extract data from.

    Returns:
        Tuple: A tuple containing match commentary, list of player objects, and other essential match information.

    This function extracts essential match data from the provided URL, including player statistics,
    match commentary, and other relevant information.

    """
    # Extract the match ID from the URL
    id_match = int(url.rsplit('-', 1)[-1])
    
    #Get the match center url from the match ( fr/previsions-pour-liverpool-nottingham-forest-1920214 -> fr/previsions-de-football/match-center/liverpool-nottingham-forest-1920214' )
    url = url.split('-')
    url = f"{url[0]}-de-football/match-center/{'-'.join(url[2:])}"
    url = "https://www.forebet.com" + url

    # Retrieve HTML code from the URL
    html_code = await get_html_code(url)

    # If HTML code retrieval fails, return
    if html_code == "":
        return

    # Parse HTML code using BeautifulSoup
    soup = BeautifulSoup(html_code, 'lxml')
    
    if soup.select_one('#st_mainData center').text == 'Match centre is unavailable for this game.':
        return [[], "", "", -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    # Get all the commentary
    commentary = soup.select_one('#st_comment').text
    
    # Get the minutes of the end of the match
    done = False
    i = 0
    regular_time, additional_time = 90, 0
    while not done and i < len(commentary.split()):
        try:
            last_commentary = commentary.split('\n\n\n')[i]
            last_minute = last_commentary.split('\n')[0]
            regular_time, additional_time = [int(objet[:-1]) for objet in last_minute.split('+')]
            done = True
        except:
            i += 1
    
    
    total_minutes_played_in_the_match = regular_time+additional_time
    
    # Get home team name and away team name
    home_team = soup.select_one("#st_hostData").contents[1].text.replace("\n", "")
    away_team = soup.select_one("#st_guestData").contents[1].text.replace("\n", "")
    
    # Initialize a list to store player objects
    list_players_of_the_match_home = []
    
    # Get home players
    st_hostPlayer_div = soup.select_one('#st_hostPlayer')
    divs = st_hostPlayer_div.find_all('div')
    for div in divs:
        # Extract player ID from div class or generate one if not found in the database
        if div['class'][0].isdigit():
            id_player = int(div['class'][0])
        else:
            done = False
            i = 0
            while not done:
                if dbManager.apply_request_to_database("SELECT COUNT(*) AS count FROM Players WHERE id_player = '{i}';")[0][0] == 0:
                    done = True
                    id_player = i
                else:
                    i += 1
        
        # Split and process player statistics
        list_stats_player = [x if x else '0' for x in div.text.split('\n')]
        
        # Create a Player object and append to the list
        list_players_of_the_match_home.append(Player(
            id_player=id_player,
            name=" ".join(list_stats_player[1].split()[1:]),
            first_name=list_stats_player[1].split()[0],
            id_match=id_match,
            sub_in=0,
            sub_out=total_minutes_played_in_the_match,
            starter=False if list_stats_player[2] == 'Sub' else True,
            football_position=list_stats_player[2],
            assists=int(list_stats_player[3]),
            goals=int(list_stats_player[4]),
            fouls_commited=int(list_stats_player[5]),
            fouls_drawn=int(list_stats_player[6]),
            offsides=int(list_stats_player[7]),
            penalty_miss=int(list_stats_player[8]),
            penalty_score=int(list_stats_player[9]),
            red_cards=int(list_stats_player[10]),
            yellow_cards=int(list_stats_player[11]),
            saves=int(list_stats_player[12]),
            shot_og=int(list_stats_player[13]),
            total_shot=int(list_stats_player[14]),
            team=home_team
        ))

        
    # Get the sub_in minute
    div_sub_in_home = soup.select_one("#st_hostSubs")
    div_of_player_sub_in = [div for div in div_sub_in_home.select('div[id]') if div.get('id') != ""]
    list_temp = []

    # Extract player ID and minute of substitution
    for div in div_of_player_sub_in:
        if div['id'].isdigit():
            id_player = int(div['id'])
        else:
            done = False
            i = 0
            while not done:
                if dbManager.apply_request_to_database("SELECT COUNT(*) AS count FROM Players WHERE id_player = '{i}';")[0][0] == 0:
                    done = True
                    id_player = i
                else:
                    i += 1
        minute_sub_in_list = div.text.split()[0][:-1].split('+')
        minute_sub_in = 0
        for minute in minute_sub_in_list:
            while not minute.isdigit():
                minute = minute[:-1]
            minute_sub_in += int(minute)
        list_temp.append((id_player, minute_sub_in))

    # Update players' substitution minute in the match
    for player_tuple_sub_in in list_temp:
        for i, player in enumerate(list_players_of_the_match_home):
            if player_tuple_sub_in[0] == player.id_player:
                list_players_of_the_match_home[i].sub_in = player_tuple_sub_in[1]


    #Get the sub_out minute
    try:
        list_temp = []
        div_home_pitch_team = soup.select_one("#st_hostTeam")
        list_divs_of_the_player_sub_out = div_home_pitch_team.select(".field_player_h > .pl_subs")
        for div in list_divs_of_the_player_sub_out:
            if type(id_player) == int or id_player.isdigit():
                id_player = int(div.parent['id'])
            else:
                done = False
                i = 0
                while not done:
                    if dbManager.apply_request_to_database("SELECT COUNT(*) AS count FROM Players WHERE id_player = '{i}';")[0][0] == 0:
                        done = True
                        id_player = i
                    else:
                        i += 1
            minute_sub_out_list = div.text.split()[-1][:-1].split('+')
            minute_sub_out = 0
            for minute in minute_sub_out_list:
                while not minute.isdigit():
                    minute = minute[:-1]
                minute_sub_out += int(minute)
            list_temp.append((id_player, minute_sub_out))
        
        # Update sub_out for the player who start 
        for player_sub_out_tuple in list_temp:
            for i, player in enumerate(list_players_of_the_match_home):
                if player_sub_out_tuple[0] == player.id_player:
                    list_players_of_the_match_home[i].sub_out = player_tuple_sub_in[1]
    except:
        pass

        
    #Get away player
    list_players_of_the_match_away = []
    st_guestPlayer_div = soup.select_one('#st_guestPlayer')
    divs = st_guestPlayer_div.find_all('div')
    for div in divs:

        # Extract player ID from div class or generate one if not found in the database
        if div['class'][0].isdigit():
            id_player = int(div['class'][0])
        else:
            done = False
            i = 0
            while not done:
                if dbManager.apply_request_to_database("SELECT COUNT(*) AS count FROM Players WHERE id_player = '{i}';")[0][0] == 0:
                    done = True
                    id_player = i
                else:
                    i += 1
# Split and process player statistics
        list_stats_player = [x if x else '0' for x in div.text.split('\n')]
        
        list_players_of_the_match_away.append(Player(
            id_player=id_player,
            name=" ".join(list_stats_player[1].split()[1:]),
            first_name=list_stats_player[1].split()[0],
            id_match=id_match,
            sub_in=0,
            sub_out=total_minutes_played_in_the_match,
            starter=False if list_stats_player[2] == 'Sub' else True,
            football_position=list_stats_player[2],
            assists=int(list_stats_player[3]),
            goals=int(list_stats_player[4]),
            fouls_commited=int(list_stats_player[5]),
            fouls_drawn=int(list_stats_player[6]),
            offsides=int(list_stats_player[7]),
            penalty_miss=int(list_stats_player[8]),
            penalty_score=int(list_stats_player[9]),
            red_cards=int(list_stats_player[10]),
            yellow_cards=int(list_stats_player[11]),
            saves=int(list_stats_player[12]),
            shot_og=int(list_stats_player[13]),
            total_shot=int(list_stats_player[14]),
            team=away_team
        ))
            
    # Get the sub_in minute
    div_sub_in_away = soup.select_one("#st_guestSubs")
    div_of_player_sub_in = [div for div in div_sub_in_away.select('div[id]') if div.get('id') != ""]
    list_temp = []

    # Extract player ID and minute of substitution
    for div in div_of_player_sub_in:
        if div['id'].isdigit():
            id_player = int(div['id'])
        else:
            done = False
            i = 0
            while not done:
                if dbManager.apply_request_to_database("SELECT COUNT(*) AS count FROM Players WHERE id_player = '{i}';")[0][0] == 0:
                    done = True
                    id_player = i
                else:
                    i += 1
        minute_sub_in_list = div.text.split()[-1][:-1].split('+')
        minute_sub_in = 0
        for minute in minute_sub_in_list:
            while not minute.isdigit():
                minute = minute[:-1]
            minute_sub_in += int(minute)
        list_temp.append((id_player, minute_sub_in))

    # Update players' substitution minute in the match
    for player_tuple_sub_in in list_temp:
        for i, player in enumerate(list_players_of_the_match_away):
            if player_tuple_sub_in[0] == player.id_player:
                list_players_of_the_match_away[i].sub_in = player_tuple_sub_in[1]


    # Get the sub_out minute
    try:
        list_temp = []
        div_away_pitch_team = soup.select_one("#st_guestTeam")
        list_divs_of_the_player_sub_out = div_away_pitch_team.select(".field_player_h > .pl_subs")
        for div in list_divs_of_the_player_sub_out:
            if div.parent['id'].isdigit():
                id_player = int(div.parent['id'])
            else:
                done = False
                i = 0
                while not done:
                    if dbManager.apply_request_to_database("SELECT COUNT(*) AS count FROM Players WHERE id_player = '{i}';")[0][0] == 0:
                        done = True
                        id_player = i
                    else:
                        i += 1
            minute_sub_out_list = div.text.split()[-1][:-1].split('+')
            minute_sub_out = 0
            for minute in minute_sub_out_list:
                while not minute.isdigit():
                    minute = minute[:-1]
                minute_sub_out += int(minute)
            list_temp.append((id_player, minute_sub_out))
    except:
        pass

    # Update sub_out for the player who start 
    for player_sub_out_tuple in list_temp:
        for i, player in enumerate(list_players_of_the_match_away):
            if player_sub_out_tuple[0] == player.id_player:
                list_players_of_the_match_away[i].sub_out = player_tuple_sub_in[1]
            
    # Concatenation of the list of the player home and away
    list_of_the_player = list_players_of_the_match_home + list_players_of_the_match_away
    
    #Get the stadium
    stadium = soup.select_one("#st_mainData div center span").text
    
    #Get all the match's statistics
    
    list_match_stats = []
    stat_divs = soup.select(".stat_cont_box")
    for i, div in enumerate(stat_divs):
        if i == 2:
            list_match_stats.append(int(div.select_one(".stat_num_left").text[:-1]))
        else:
            list_match_stats.append(int(div.select_one(".stat_num_left").text) if div.select_one('.stat_num_left').text != "" else 0)
        
        if i == 2:
            list_match_stats.append(int(div.select_one(".stat_num_right").text[:-1]))
        else:
            list_match_stats.append(int(div.select_one(".stat_num_right").text) if div.select_one(".stat_num_right").text != "" else 0)
    if stat_divs == []:
        list_match_stats = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    shot_home, shot_away, shot_og_home, shot_og_away, possession_home, possession_away, corner_home, corner_away, offside_home, offside_away, fouls_home, fouls_away, saves_home, saves_away = list_match_stats
    
    return list_of_the_player, stadium, commentary, shot_home, shot_away, shot_og_home, shot_og_away, possession_home, possession_away, corner_home, corner_away, offside_home, offside_away, fouls_home, fouls_away, saves_home, saves_away


async def main():
    temps_debut = time.time()
    for league in list_league_to_add:
        await get_all_matchs_league_data_brut(league)
    

    temps_ecoule = time.time() - temps_debut
    print("Temps écoulé:", temps_ecoule, "secondes")
#League(ID_League='BL20152016', isdone=True, forebet_url="https://www.forebet.com/fr/previsions-de-football-pour-allemagne/bundesliga/results/2015-2016", time_zone_difference=0),
                      
list_league_to_add = [League(ID_League='LIGA20232024', isdone=False, forebet_url="https://www.forebet.com/fr/previsions-de-football-pour-espagne/primera-division/results/2023-2024", time_zone_difference=0),
                      League(ID_League='L120232024', isdone=False, forebet_url="https://www.forebet.com/fr/previsions-de-football-pour-france/league1/results/2023-2024", time_zone_difference=0),
                      League(ID_League='BL20232024', isdone=False, forebet_url="https://www.forebet.com/fr/previsions-de-football-pour-allemagne/bundesliga/results/2023-2024", time_zone_difference=0),
                      League(ID_League='SerieA20232024', isdone=False, forebet_url="https://www.forebet.com/fr/previsions-de-football-pour-italie/seriea/results/2023-2024", time_zone_difference=0)]
# Run the main function asynchronously
asyncio.run(main())

""" 
All the league object added to the database are:
    PL from 2015 to 2023, LIGA from 2015 to 2023, L1 from 2015 to 2023, BL from 2015 to 2023, SerieA from 2015 to 2023
    
"""