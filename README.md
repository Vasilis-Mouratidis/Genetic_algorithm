The project that is given, informs us that there are a number of wasp nests in an attic. The nests have a different number of wasps in them.
The table consists of the coordinates of the nests and the number of wasps in them.
We are given three(3) "wasp-bombs". It is known that these three bombs cannot kill all 100% of the wasps.
I am asked to find the best possible bomb placement in order to kill the biggest possible percentage of of the wasps, through the use of evolutionary algorithms.
This evolutionary algorithm follows these steps:
	1. place the bombs randomly 
	2. find the killing percentage(each placement of the three bombs constitutes a generation)
		2.1. if the killing percentage of the last 100 generations hasnt been improved, the itteration stops, and we keep that best solution.
	3. this solution, in order to become better, undergoes a mutation which changes the bomb coordinates and then it tests if this solution is better from the previously best one.
The project in called "Genetic_algorithm".
