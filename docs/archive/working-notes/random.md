How many different runtimes support needed?:
as this is a research project, which will get into production in fturue, by real user:
but for focusing on current architecture:
- pure nsga only (baseline GA)
- nsga + repairs
- nsga + ga specific  repairs and those small heuristics
- nsga + repairs + local search
- nsga + rl + (repairs + local search) but rl guided repari and search
- nsga + round robin heuristic selector from toolbox + repairs + local search
- nsga + ? what else? [constraint weight tuned by rl?]

and what types of comparsion study ? even not here you suggest? or can i remove any of above runtimes?
- i also need to modify the main.py and other codebased and ig should create wrappers around config to make upperconfigs for easy switching between runtimes. What do you think?

and how are output handled in these kind of experiments? my approach is slightly fuckedup right? how can i enhance it? How can i manage outputs from different experiments loggers etc. and how to configure it as champ level researcher and product designer. 



new rpompt
Look at the implementation roadmap of : 11-advanced-techniques-suggest:?
further topics are on same folder:
11-advanced-techniques-suggest/
make a list of things analyzing dependencies and everything: to implement at Phase 1:: 