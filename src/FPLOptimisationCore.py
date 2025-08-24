######################################################################
# USER INPUT VARIABLES
######################################################################

Default_Data_Path =  "data\Start_2025-6_players_raw.csv" #Expected to be previous seasons' data

#END
######################################################################
import pandas as pd
from pulp import *

class Players: 

    def __init__(self, path=Default_Data_Path): 
        ### Load and preprocess data
        self.data = pd.read_csv(path)
        self.data = self.preprocess_data(self.data)

    def preprocess_data(self, df):
        df = df.set_index('id')

        df['total_games'] = df['minutes'] / 90
        df = df[['first_name', 'second_name', "web_name", 'element_type', 'team', 'now_cost', 'minutes',   
                 'chance_of_playing_next_round','total_games', 'total_points']]
        #Remove players with no chance of playing by multiplying points by chance of playing
        df['chance_of_playing_next_round'] = pd.to_numeric(df['chance_of_playing_next_round'], errors='coerce')
        df['chance_of_playing_next_round'] = df['chance_of_playing_next_round'].fillna(100) #treat Na as 100% chance of playing
        print(df.loc[467][["second_name","total_points", "chance_of_playing_next_round"]])
        df["total_points"] = df["total_points"] * df["chance_of_playing_next_round"]/100
        print(df.loc[467][["second_name","total_points", "chance_of_playing_next_round"]])
        

        ##legibility
        #Rename columns
        df = df.rename(columns={'now_cost': 'cost', 'element_type': 'position', 'total_points': 'points'})
        
        
        # Create dummy variables (a single 0/1 column for each team and position) and glue them to the original dataframe
        df['team'] = df['team'].astype('category')
        df['position'] = df['position'].astype('category')
        dummies = pd.get_dummies(df[['team', 'position']])
        dummies = dummies.rename(columns={
            'position_1': 'keeper',
            'position_2': 'defender',
            'position_3': 'midfielder',
            'position_4': 'forward'
        })
        df = pd.concat([df, dummies], axis=1)

        #add position names
        # df = df.sort_values(by='position')
        df["position_name"] = df.apply(
            lambda row : next(position for position in ["keeper", "defender", "midfielder", "forward"] if row[position])
            ,axis = 1
        )

        return df
    
    def NormaliseCurrentData(self, gameweek): 
        #Add (last seasons' total points * player's current chance of playing) to current points
        #Normalise to 38 gameweeks so points represents expected points for the whole season
        previous_season_df = Players().data
        current_play_chance = self.data["chance_of_playing_next_round"]/100 
        self.data["points"] += current_play_chance * (
            previous_season_df["points"]    #player id is used as index, so safe to naively add together 
            .reindex(self.data.index, fill_value=0) #Add 0 if index not in current data
            ) 
        self.data["points"] *= 38 / (38 + gameweek)
        print (self.data["points"])
        return self




class Parameters:
    ### Parameters for the problem`
    squad_size = 15
    starters = 11
    subs = squad_size - starters
    tot_budget = 1000
    budget_split = range(700, tot_budget - 125, 5)
    #Must have this many players of each position 
    #TODO refactor any uses of tot_xxx to use positions_totals
    tot_keepers = 2
    tot_defenders = 5
    tot_midfielders = 5
    tot_forwards = 3
    positions_totals = {
            'keepers': 2,
            'defenders': 5,
            'midfielders': 5,
            'forwards': 3
            }
    per_team = 3 #Max 3 players from each team in the league can be in squad
    tot_gameweeks = 38
    subs_weighting = 0.1 # Weight subs points to significantly less than starters point. Ensures budget isn't wasted on subs over starters, but forces model to pick best performing subs (as opposed to completely ignoring points contributions from subs)
    subs_games_threshold = 0.5 #Proportion of games needed to consider a player for subs
    subs_min_games = round(subs_games_threshold * tot_gameweeks)
    starters_positions_bounds = {
        'lower': {
            'keepers': 1,
            'defenders': 3,
            'midfielders': 2,
            'forwards': 1
        },
        'upper': {
            'keepers': 1,
            'defenders': 5,
            'midfielders': 5,
            'forwards': 3
        }
    }
    def __init__ (self, df): 
        self.teams = df['team'].unique()
        self.teams_dict = dict(zip([f'team_{t}' for t in self.teams], [Parameters.per_team for i in self.teams]))

#Honestly thought we'd need these classes, turns out no.
class StartingLineup_Parameters (Parameters): 
    #Override base Parameters attributes as needed
    pass
class Transfer_Parameters (Parameters): 
    #Override base Parameters attributes as needed
    pass


#Maximise points for a given number of players. Intended to be used independently for starters and subs
def max_points_model(fn_df #dataframe of players + associated data to chose from
                     , budget #in £xe4 (1000 = £10m), because that's what the dataset uses 
                     , num_players #In practice should be either Parameters.starters or Parameters.subs
                     , positions_d #Dict of bounds on no. players per position
                     , teams_d #dict of teams and upper bound of selectable players per team (3 per team overall, but needs to be treated carefully when optimising subs seperately to starters ) 
                     ):
    # Define the optimization problem
    model = LpProblem("FantasyFootball", LpMaximize)

    # Define decision variables
    decision_vars = LpVariable.dicts("player", fn_df.index, 0, 1, LpInteger)

    #helper
    def SumColumnTimesDecisionVars(column_name: str):
        return sum(fn_df.loc[i, column_name] * decision_vars[i] for i in fn_df.index)

    # Objective function (maximize total points)
    model += SumColumnTimesDecisionVars('points')

    # Constraints 
    model += sum(decision_vars[i] for i in fn_df.index) == num_players, "NumberOfPlayers"
    model += SumColumnTimesDecisionVars('cost') <= budget, "Budget"
    model += SumColumnTimesDecisionVars('keeper') <= positions_d['upper']['keepers'], "Keepers upper"
    model += SumColumnTimesDecisionVars('keeper') >= positions_d['lower']['keepers'], "Keepers lower"
    model += SumColumnTimesDecisionVars('defender') <= positions_d['upper']['defenders'], "Defenders upper"
    model += SumColumnTimesDecisionVars('defender') >= positions_d['lower']['defenders'], "Defenders lower"
    model += SumColumnTimesDecisionVars('midfielder') <= positions_d['upper']['midfielders'], "Midfielders upper"
    model += SumColumnTimesDecisionVars('midfielder') >= positions_d['lower']['midfielders'], "Midfielders lower"
    model += SumColumnTimesDecisionVars('forward') <= positions_d['upper']['forwards'], "Forwards upper"
    model += SumColumnTimesDecisionVars('forward') >= positions_d['lower']['forwards'], "Forwards lower"

    for team, places_left in teams_d.items():
        model += SumColumnTimesDecisionVars(team) <= places_left, team

    # Solve the model
    model.solve(PULP_CBC_CMD(msg=False))
    print(budget, ':')
    print(LpStatus[model.status])
    model_success = LpStatus[model.status]

    ### Calculate and print summary
    # for i in fn_df.index:
    #     if decision_vars[i].varValue > 0:
    #         print(fn_df.loc[i, 'points'] * decision_vars[i].varValue )
    total_points = sum(fn_df.loc[i, 'points'] * decision_vars[i].varValue for i in fn_df.index)
    total_cost = sum(fn_df.loc[i, 'cost'] * decision_vars[i].varValue for i in fn_df.index)
    selected_players = [i for i in fn_df.index if decision_vars[i].varValue == 1]
    # selected_players_names = [f"{fn_df.loc[i, 'first_name']} {fn_df.loc[i, 'second_name']}" for i in fn_df.index if
    #                           decision_vars[i].varValue == 1]
    selected_players_fn_df = fn_df.loc[selected_players]

    # Categorize players by position
    model_keepers = selected_players_fn_df[selected_players_fn_df['keeper'] == 1].index.tolist()
    model_defenders = selected_players_fn_df[selected_players_fn_df['defender'] == 1].index.tolist()
    model_midfielders = selected_players_fn_df[selected_players_fn_df['midfielder'] == 1].index.tolist()
    model_forwards = selected_players_fn_df[selected_players_fn_df['forward'] == 1].index.tolist()
    model_positions = {
        'keepers': model_keepers,
        'defenders': model_defenders,
        'midfielders': model_midfielders,
        'forwards': model_forwards
    }

    # Categorize players by team
    model_teams = {}
    for team in teams_d:
        model_teams[team] = len(selected_players_fn_df[selected_players_fn_df[team] == 1].index.tolist())

    return total_points, total_cost, model_positions, model_teams, selected_players, model_success

def remove_players(old_df, remove_lst):
    new_df = old_df.drop(index=remove_lst)
    return new_df

def max_points_transfers_with_subs_model( 
#Unlike max_points_model, for transfers team need to be treated together as
#   - Starting lineup has different bounds on positions numbers than the overrall team (e.g. need 3 fwds total even if only one on the pitch). So trying to transfer only the starting lineup gives infeasible solutions as the existing subs aren't necessarily compatible (and constraining the transfers to the currently used formation would be silly)
#   -  Trying to optimise subs seperately to starters would be inefficient as they don't contribute much to points but could be useful to sell in order to make up the budget to afford a starting player.
        all_players_df, #dataframe of players + associated data to chose from
        current_players_df, #ids of players currently in squad
        budget,  #in £xe4 (1000 = £10m), because that's what the dataset uses 
        parameters: Parameters,
        gameweek, #Up to 38, needed to calc expected points accurately
        NumFreeTransfers = 1 # Number of free transfers, default to 1 but these stack up to 5 if not used in previous gameweeks
        ):
    # Define the optimization problem
    model = LpProblem("FantasyFootball", LpMaximize)

    # Define decision variables
    # Need to constrain starters to be not in subs
    starters = LpVariable.dicts("starter", all_players_df.index, 0, 1, LpInteger)
    subs = LpVariable.dicts("sub", all_players_df.index, 0, 1, LpInteger)

    #Other variables
    bounded_transfer_points_deducted = LpVariable("min_transfer_points_deducted", lowBound=0, cat = "Integer") #only used to prevent transfer penalty from going negative

    #Helper functions for writing constraints
    def players(): return { id : starters[id] + subs[id] for id in starters }
    def Total(column_name: str, decisions_d: LpVariable.dicts):
        return sum(all_players_df.loc[i, column_name] * decisions_d[i] for i in all_players_df.index)
    def players_sold_cost(): return sum(
        all_players_df.loc[id, 'cost'] 
        * (1 - players()[id]) #  == sum if player not in decisions
        for id in all_players_df.index 
        if id in current_players_ids
        )
    def players_bought_cost(): return sum(
        all_players_df.loc[id, 'cost'] 
        * players()[id] 
        for id in all_players_df.index 
        if id not in current_players_ids
        )
    def numTransfers(): return sum(1 - players()[id] for id in current_players_ids) #number of players sold, used to calc transfer points deducted  
    def totalPoints(): # sum total points scaled by proportion of gameweeks, with subs weighted much less than starters
        return (
        ( Total("points", starters) + Total("points", subs) * parameters.subs_weighting )
            * gameweek_scaling 
        - bounded_transfer_points_deducted
        )
    ObjectiveFunction = totalPoints

    #helper variables
    def players_varValue(id): return starters[id].varValue + subs[id].varValue
    current_players_ids = current_players_df['id'].tolist()  #ids of current players
    current_starters_ids = current_players_df[current_players_df["role"] == "starter"]['id'].tolist()  
    current_subs_ids = current_players_df[current_players_df["role"] == "sub"]['id'].tolist()  
    num_players = len(current_players_ids) #this was an arg in the starting lineup model, but can infer it here
    gameweek_scaling = (parameters.tot_gameweeks - gameweek) / parameters.tot_gameweeks
    positions_d = parameters.starters_positions_bounds, #Dict of bounds on no. players per position
    teams_d = parameters.teams_dict  #dict of teams and upper bound of selectable players per team (3 per team overall, but needs to be treated carefully when optimising subs seperately to starters )
 
    
    print("\nSanity Checks:")
    print(f"Budget: {budget}")
    print(f"Current players: {current_players_ids}")
    print(f"Number of players: {num_players}")
    print(f"Gameweek scaling: {gameweek_scaling}")
    print()

    # Objective function (maximize total points)
    model += ObjectiveFunction(), "TotalPoints"  

    ### Constraints ###

    # Number of players 
    model += ( sum(players()[i] for i in all_players_df.index) 
              == num_players, "NumberOfPlayers" )
    model += ( sum(starters[i] for i in all_players_df.index) 
              == parameters.starters, "NumberOfStarters" )
    model += ( sum(subs[i] for i in all_players_df.index) 
              == parameters.subs, "NumberOfSubs" )

    for id in all_players_df.index:
        #Starters can't appear in subs
        model += starters[id] + subs[id] <= 1, f"StartersAreNotSubs{id}"
        #each sub must play more than the threshold
        model += all_players_df.loc[id, "total_games"] >= parameters.subs_games_threshold * subs[id], f"SubsMinGames_{id}"

    #No. starters & players per Position 
    for position in ['keeper', 'defender', 'midfielder', 'forward']:
        # starter bounds
        model += ( Total(position, starters) 
                  <= parameters.starters_positions_bounds['upper'][f'{position}s'], f"starting {position}s upper" )
        model += ( Total(position, starters) 
                  >= parameters.starters_positions_bounds['lower'][f'{position}s'], f"starting {position}s lower" )
        # squad totals
        model += ( Total(position, players()) 
                  == parameters.positions_totals[f'{position}s'], f"squad {position}s total" )

    # No. players from each Team
    for team, places_left in teams_d.items():
        model += Total(team, players()) <= places_left, team

    #players bought <= budget + players sold
    model += players_bought_cost() <= players_sold_cost() , "Budget" 
    
    #Transfer penalty must be >= 0
    model += ( bounded_transfer_points_deducted 
              >= 4 * (numTransfers() - NumFreeTransfers), "transferPenaltyisPositive" )

    ### Solve the model
    # model.writeMPS("debug.mps") #to inspect mps file if internal solver error occurs
    model.solve(PULP_CBC_CMD(msg=False))
    print(LpStatus[model.status])
    model_success = LpStatus[model.status]


    #helper functions for outputting results 
    #Mostly clones of helper functions above but using decision_vars.varValue. Is there a nicer way to do that?
    def Total_result(column_name: str, decisions_d: LpVariable.dicts):
        return sum(all_players_df.loc[i, column_name] * decisions_d[i].varValue for i in all_players_df.index)
    
    players_sold_ids_result = [id for id in current_players_ids if players_varValue(id) == 0]
    players_sold_cost_result = 0 + sum(all_players_df.loc[id, 'cost'] for id in players_sold_ids_result)
    players_bought_ids_result = [id for id in all_players_df.index if players_varValue(id) == 1 and id not in current_players_ids]
    players_bought_cost_result = sum(all_players_df.loc[id, 'cost'] for id in players_bought_ids_result)
    numTransfers_result = sum(1 - players_varValue(id) for id in current_players_ids) #number of players sold, used to calc transfer points deducted  
    transfer_points_deducted_result =  max(4 * (numTransfers_result - NumFreeTransfers), 0)
    
    subs_contribution_to_total_points = 0 #Ignoring subs points contributions seems most sensible for results
    #Could set this to parameters.subs_weighting, but that's not a well motivated number (it's really just used to ensure the model prioritises subs with more points).  
    def totalPoints_result() -> int :
        return (
                Total_result("points", starters)
            +   Total_result("points", subs) * subs_contribution_to_total_points #Subs
            ) * gameweek_scaling - transfer_points_deducted_result
        
    def current_players_TotalPoints() -> int :
        return (
                sum(all_players_df.loc[id, 'points'] for id in current_starters_ids)
            +   sum(all_players_df.loc[id, 'points'] for id in current_subs_ids) * subs_contribution_to_total_points 
        ) * gameweek_scaling

    print(f"numTransfers_result: {numTransfers_result}")
    print(f"transfer_points_deducted_result: {transfer_points_deducted_result}")
    print(f"totalPoints_result: {totalPoints_result}")
    print(f"new_players_ids:{[i for i in all_players_df.index if players_varValue(i) == 1]}")
    print(f"current_players_ids:{current_players_ids}")
    print(f"current_players_TotalPoints: {current_players_TotalPoints}")
    


    ### Calculate and print summary
    total_points = totalPoints_result()
    total_cost = players_bought_cost_result
    budget_remaining = budget + players_sold_cost_result - total_cost
    selected_players = [i for i in all_players_df.index if players_varValue(i) == 1]
    selected_starters = [i for i in all_players_df.index if starters[i].varValue == 1]
    selected_subs = [i for i in all_players_df.index if subs[i].varValue == 1]
    
    numTransfers = len(players_sold_ids_result)
    transfer_points_deducted = 4 * max(numTransfers - NumFreeTransfers, 0)



    # selected_players_names = [f"{fn_df.loc[i, 'first_name']} {fn_df.loc[i, 'second_name']}" for i in fn_df.index if
    #                           decision_vars[i].varValue == 1]
    selected_players_fn_df = all_players_df.loc[selected_players]

    # Categorize players by position
    model_keepers = selected_players_fn_df[selected_players_fn_df['keeper'] == 1].index.tolist()
    model_defenders = selected_players_fn_df[selected_players_fn_df['defender'] == 1].index.tolist()
    model_midfielders = selected_players_fn_df[selected_players_fn_df['midfielder'] == 1].index.tolist()
    model_forwards = selected_players_fn_df[selected_players_fn_df['forward'] == 1].index.tolist()
    model_positions = {
        'keepers': model_keepers,
        'defenders': model_defenders,
        'midfielders': model_midfielders,
        'forwards': model_forwards
    }

    # Categorize players by team
    model_teams = {}
    for team in teams_d:
        model_teams[team] = len(selected_players_fn_df[selected_players_fn_df[team] == 1].index.tolist())

    return {
        'total_points': total_points,
        'total_cost': total_cost,
        'budget_remaining': budget_remaining,
        'model_positions': model_positions,
        'model_teams': model_teams,
        'model_success': model_success,
        'selected_players': selected_players,
        'selected_starters': selected_starters,
        'selected_subs': selected_subs,
        'players_sold_ids': players_sold_ids_result,
        'players_sold_cost': players_sold_cost_result,
        'players_bought_ids': players_bought_ids_result,
        'players_bought_cost': players_bought_cost_result,
        'numTransfers': numTransfers,
        'transfer_points_deducted': transfer_points_deducted,
        'current_players_total_points':current_players_TotalPoints()
    }



