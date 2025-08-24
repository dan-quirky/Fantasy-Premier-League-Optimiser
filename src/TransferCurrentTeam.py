### Optimise a Fantasy Premier League team by transferring from a given starting team, including substitutes

#Expects
# - players_raw.csv data from https://github.com/vaastav/Fantasy-Premier-League
# - csv of the current team data (this is output by the StartingLineUpWithSubs & TransferTeam & can be manually edited if needed)

# TODO add a CreateCurrentTeamCSV.py for cases where team doesn't match a previous output
######################################################################
# USER INPUT VARIABLES
######################################################################
#update for each gameweek
current_gameweek = 1 # Used to normalise the expected points data, and scale total points calculation in the objective function by the proportion of game weeks remaining (i.e. the transfer penalty becomes more significant the fewer weeks are left in the season )
budget = 0 #in millions of £
numFreeTransfers = 1
current_team_path = "current_team.csv"
data_path = "data\players_raw_gw1.csv"

#END
######################################################################
from FPLOptimisationCore import Players, Transfer_Parameters, max_points_transfers_with_subs_model
from datetime import datetime 
import os
import pandas as pd


budget = budget * 10 #convert budget to weird units data uses

all_players_df = (
    Players("data\players_raw_gw1.csv")
    .NormaliseCurrentData(current_gameweek)
    .data
    )
p = Transfer_Parameters(all_players_df)

current_team_df = pd.read_csv(current_team_path)

  
### Run the transfer optimisation
result = max_points_transfers_with_subs_model(
            all_players_df,
            current_team_df,
            budget,
            p,
            current_gameweek,
            numFreeTransfers)


# print(f"Result: {result}")
# print(all_players_df.loc[result["selected_players"]]
#       [['first_name', 'second_name', 'position', 'team', 'cost', 'points']])
# print()

#helper funcs & vars to print out results
def sumCost(df): return formatCost(sum(df["cost"]))
def formatCost(cost): return '{:,}'.format(int(cost* 1e5))

def playersInfo(position: str, df): 
    ids = df.index
    if ids in df.loc[result["model_positions"][position]].index:
        print (df.loc[id][['first_name', 'second_name', 'cost', 'points']])




###Print out results
print("\n\n")

players_sold_df = all_players_df.loc[result["players_sold_ids"]]
print(f"Sold {", ".join(players_sold_df["web_name"].to_list())} for £{sumCost(players_sold_df)}")
print('and')
players_bought_df = all_players_df.loc[result["players_bought_ids"]]
print(f"Bought {", ".join(players_bought_df["web_name"].to_list())} for £{sumCost(players_bought_df)}")
print(f"Budget remaining: £{formatCost(result["budget_remaining"])}")
print("\n\n")

players_df = starters_df = all_players_df.loc[result["selected_players"]]
players_df['role'] = ['starter' if i in result["selected_starters"] else 'sub' for i in players_df.index]
players_df = players_df.sort_values(by=["role",'position'])
players_df = players_df[["position_name", "role", 'web_name', 'cost', 'points']]
print("New Squad:")
print(players_df)
print("\n\n")

print(f"New lineup's expected points, including transfer penalties: {int(result["total_points"])}")
print(f"(Previous lineup's expected points: {int(result["current_players_total_points"])})")
print("\n\n")

output_name = f"Transfers_in_gameweek_{current_gameweek}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
output_path = os.path.join("output",output_name)
print(f"Saving as \"{output_name}\"")
players_df.to_csv(output_path)


