### Optimise a Fantasy Premier League starting team including substitutes
#Expects players_raw.csv data from https://github.com/vaastav/Fantasy-Premier-League
######################################################################
# USER INPUT VARIABLES
######################################################################

#Nothing to set here, default data path should be configured in FPLOptimisationCore

#END
######################################################################
from FPLOptimisationCore import Players, StartingLineup_Parameters, max_points_model, remove_players
from datetime import datetime 
import os
import pandas as pd

df = Players().data
p = StartingLineup_Parameters(df)

#Loop over different budgets to find best starters team with acceptable substitutes 
results = []
for squad_budget in p.budget_split:
    # 1: Find best starters for given budget
    starters_df = df
    starters_points, starters_cost, starters_positions, starters_teams, starters_indices, starters_success = \
        max_points_model(
            starters_df,
            squad_budget,
            p.starters,
            p.starters_positions_bounds,
            p.teams_dict)
    # 2. Find best subs with remaining budget
    # Filter out players not suitable for subs
    subs_df = remove_players(starters_df, starters_indices)
    subs_df = subs_df[subs_df['total_games'] >= p.subs_min_games]
    subs_budget = p.tot_budget - squad_budget
    subs_positions_bounds = {
        'lower': {
            'keepers': p.tot_keepers - len(starters_positions['keepers']),
            'defenders': p.tot_defenders - len(starters_positions['defenders']),
            'midfielders': p.tot_midfielders - len(starters_positions['midfielders']),
            'forwards': p.tot_forwards - len(starters_positions['forwards'])
        },
        'upper': {
            'keepers': p.tot_keepers - len(starters_positions['keepers']),
            'defenders': p.tot_defenders - len(starters_positions['defenders']),
            'midfielders': p.tot_midfielders - len(starters_positions['midfielders']),
            'forwards': p.tot_forwards - len(starters_positions['forwards'])
        }
    }
    subs_teams_dict = {}
    for team, places_left in p.teams_dict.items():
        subs_teams_dict[team] = places_left - starters_teams[team]

    subs_points, subs_cost, subs_positions, subs_teams, subs_indices, subs_success = \
        max_points_model(
            subs_df,
            subs_budget,
            p.subs,
            subs_positions_bounds,
            subs_teams_dict)

    total_points = starters_points + 0.1 * subs_points #this isn't well motivated, but the subs need to have some contribution to toal points or the model ignores them
    total_cost = starters_cost + subs_cost
    complete_team_indices = starters_indices + subs_indices

    result = {
        'budget_split': squad_budget,
        'starters_points': starters_points,
        'starters_cost': starters_cost,
        'starters_indices': starters_indices,
        'starters_success': starters_success,
        'subs_points': subs_points,
        'subs_cost': subs_cost,
        'subs_indices': subs_indices,
        'subs_success': subs_success,
        'total_points': total_points,
        'total_cost': total_cost,
        'complete_team_indices': complete_team_indices
    }
    #Write out results
    results.append(result)

results_df = pd.DataFrame(results)

print(results_df)

#Get the result that returned highest points
results_df_success = results_df[
    (results_df['starters_success'] == 'Optimal') & (results_df['subs_success'] == 'Optimal')]
optimal_result = results_df_success.loc[results_df_success['total_points'].idxmax()]

# Print the optimal budget split and its corresponding total points
print(f"Optimal Budget Split: {optimal_result['budget_split']}")
print(f"Maximum Total Points: {optimal_result['total_points']}")
print(f"Starters Cost: {optimal_result['starters_cost']}")
print(f"Substitutes Cost: {optimal_result['subs_cost']}")
print(f"Total Cost: {optimal_result['total_cost']}")

optimal_starters_indices = optimal_result['starters_indices']
optimal_subs_indices = optimal_result['subs_indices']
optimal_complete_team_indices = optimal_result['complete_team_indices']

optimal_team = df.loc[optimal_complete_team_indices]
optimal_team['role'] = ['starter' if i in optimal_starters_indices else 'sub' for i in optimal_team.index]
optimal_team = optimal_team.sort_values(by=['position', 'role'], ascending=[True, True])
print("Optimal Team:")
print(optimal_team[['first_name', 'second_name', 'position', 'team', 'cost', 'points', 'role']])

#output data
output_name = f"Starting_Squad_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
output_path = os.path.join("output",output_name)
print(f"Saving as \"{output_name}\"")
optimal_team.to_csv(output_path)


# Plot the total points vs. budget split
# results_df_success.plot(x='budget_split', y='total_points', legend=False)
# plt.xlabel('Budget Split for Starters')
# plt.ylabel('Total Points')
# plt.show()
