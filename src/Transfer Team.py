### Optimise Fantasy Premier League starting team with substitutes

from FPLOpt_Core import *
from datetime import datetime
import os

# all_players_df = Players("TESTONLY- Injured_Starting_Lineup_set_to_0_2025-6_Start_players_raw_Total_Games_Weighted_by_Chance - Copy.csv").data
# all_players_df = Players().data

# all_players_previous_season_df = Players().data
# all_players_df = Players("data\players_raw_gw1.csv").data
# for id in all_players_df.index :
#     if id in all_players_previous_season_df.index:
#         #add previous seasons points, normalised to current seasons chance of playing
#         all_players_df.loc[id,"points"] += all_players_previous_season_df.loc[id,"points"] * all_players_df.loc[id,"chance_of_playing_next_round"]/100 
#         all_players_df["points"] *= 38/39
# print(all_players_df["points"])

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
budget = budget * 10 #convert budget to weird units data uses

all_players_df = (
    Players("data\players_raw_gw1.csv")
    .NormaliseCurrentData(current_gameweek)
    .data
    )
p = Transfer_Parameters(all_players_df)

#Run transfers model on current lineup for to find best transfers for current weeks data (different prices, points, chance of playing etc)
current_team_df = pd.read_csv(current_team_path)

# result = max_points_transfers_model(
#             all_players_df,
#             current_team_df,
#             budget,
#             p,
#             current_gameweek,
#             numFreeTransfers)
    

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


#new lineups expected points


    #Write out results
    # results.append(result)

# optimal_result = pd.DataFrame(result)

# # print(results_df)

# # #Get the result that returned highest points
# # results_df_success = results_df[
# #     (results_df['starters_success'] == 'Optimal') & (results_df['subs_success'] == 'Optimal')]
# # optimal_result = results_df_success.loc[results_df_success['total_points'].idxmax()]

# # Print the optimal budget split and its corresponding total points
# print(f"Maximum Total Points: {optimal_result['total_points']}")
# print(f"Total Cost: {optimal_result['total_cost']}")

# optimal_starters_indices = optimal_result['starters_indices']
# optimal_subs_indices = optimal_result['subs_indices']
# optimal_complete_team_indices = optimal_result['complete_team_indices']

# optimal_team = df.loc[optimal_complete_team_indices]
# optimal_team['role'] = ['starter' if i in optimal_starters_indices else 'sub' for i in optimal_team.index]
# optimal_team = optimal_team.sort_values(by=['position', 'role'], ascending=[True, True])
# print("Optimal Team:")
# print(optimal_team[['first_name', 'second_name', 'position', 'team', 'cost', 'points', 'role']])
# optimal_team.to_csv("starting_team.csv")

