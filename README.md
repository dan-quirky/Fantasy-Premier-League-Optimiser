# Fantasy Premier League Optimiser
## What does this do?
Uses linear programming to:
- Create an optimal starting fantasy team based on the previous Premier League season's data
- Optimise your transfers in a given game week based on current and past seasons' data
## What doesn't this do?
Teach you anything about football (I see this as a bonus)
## How to use
1) Install necessary modules with `pip install -r /path/to/requirements.txt`. 
2) Download latest players_raw.csv from [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League/tree/master/data )
	- Alternatively, the repo includes sample data from the 2025-26 pre-season and gameweek 1
3) Set user variables in "FPLOptimisationCore.py"
4) Set user variables in "CreateStartingTeam.py" or "TransferCurrentTeam.py" as required, and run.

## Credit
- Data is from vaastav's [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League/tree/master/data) 
- [PuLP](https://coin-or.github.io/pulp/) is used to set up and solve optimisations as linear programming problems
- Initial implementation of the starting line-up optimisation was created by someone who did not want to be credited
- Anything else is my fault
