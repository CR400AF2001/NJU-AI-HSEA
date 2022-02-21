# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
A* search , run the following command:

> python pacman.py -l smallmaze -p EvolutionSearchAgent

"""
import itertools
import random

from game import Directions
from game import Agent
from game import Actions
from searchAgents import SearchAgent, PositionSearchProblem, FoodSearchProblem
import util
import time
import search


#########################################################
# This portion is written for you, but will only work   #
#     after you fill in parts of EvolutionSearchAgent   #
#########################################################

positionDict = {}

def positionHeuristic(state, times):
    '''
    A simple heuristic function for evaluate the fitness in task 1
    :param state:
    :return:
    '''
    if state in positionDict.keys():
        score = abs(state[0] - 1) + abs(state[1] - 1) + times * positionDict[state]
    else:
        score = abs(state[0] - 1) + abs(state[1] - 1)
    return score


class EvolutionSearchAgent():
    def __init__(self, type='PositionSearchProblem', actionDim=10):
        '''
        This is the EvolutionSearchAgent, you should firstly finish the search.evolutionSearch
        '''
        self.searchType = globals()[type]
        self.actionDim = actionDim  # dim of individuals
        self.T = 20  # iterations for evolution
        self.popSize = 20  # number of individuals in the population
        self.population = None
        self.callFitness = 0
        self.mutateProb = 0.2
        self.problem = None
        self.offspringNum = 10
        self.times = 100

    def getFitness(self, state):
        '''
        evaluate the individuals
        note that you should record the number of using getFitness, and report it at the end of the task.
        :param state:
        :return:
        '''
        self.callFitness += 1

        return positionHeuristic(state, self.times)

    def mutation(self, individual):
        first = 0
        for i in range(1, self.actionDim + 1):
            if random.random() < self.mutateProb:
                successors = self.problem.getSuccessors(individual[i - 1][0])
                otherAction = []
                for j in range(0, len(successors)):
                    if successors[j][1] != individual[i][1]:
                        otherAction.append([successors[j][0], successors[j][1]])
                if len(otherAction) != 0:
                    next = random.choice(otherAction)
                    individual[i] = [next[0], next[1]]
                if first == 0:
                    first = i
        if first == 0:
            return individual
        else:
            return self.checkValid(individual, first + 1)

    def selectParents(self):
        self.population = sorted(self.population, key=lambda x: x[1])
        return self.rouletteWheel()

    def rouletteWheel(self):
        sumFitness = 0
        choosen = []
        if self.population[0][1] != 0:
            for i in range(0, self.popSize):
                sumFitness += 1.0 / self.population[i][1]
            for _ in range(0, 2):
                temp = 0
                randomNum = random.uniform(0, sumFitness)
                for i in range(self.popSize - 1, -1, -1):
                    temp += 1.0 / self.population[i][1]
                    if randomNum <= temp:
                        choosen.append(self.population[i])
                        break
        else:
            for i in range(0, self.popSize):
                sumFitness += 1.0 / (self.population[i][1] + 1)
            for _ in range(0, 2):
                temp = 0
                randomNum = random.uniform(0, sumFitness)
                for i in range(self.popSize - 1, -1, -1):
                    temp += 1.0 / (self.population[i][1] + 1)
                    if randomNum <= temp:
                        choosen.append(self.population[i])
                        break
        return choosen[0], choosen[1]

    def crossover(self, parent1, parent2):
        crossPoint = random.randint(2, len(parent1[0]) - 1)
        offspring1 = parent1[0][:crossPoint] + parent2[0][crossPoint:]
        offspring2 = parent2[0][:crossPoint] + parent1[0][crossPoint:]
        return self.checkValid(offspring1, crossPoint), self.checkValid(offspring2, crossPoint)

    def checkValid(self, individual, start):
        for i in range(start, self.actionDim + 1):
            successors = self.problem.getSuccessors(individual[i - 1][0])
            found = False
            for s in successors:
                if individual[i][1] == s[1]:
                    individual[i] = [s[0], s[1]]
                    found = True
                    break
            if not found:
                next = random.choice(successors)
                individual[i] = [next[0], next[1]]
        return individual

    def generateLegalActions(self):
        '''
        generate the individuals with legal actions
        :return:
        '''
        self.population = []
        state = self.problem.getStartState()

        for i in range(0, self.popSize):
            individual = []
            currentState = state
            individual.append([state, Directions.STOP])
            for j in range(1, self.actionDim + 1):
                successors = self.problem.getSuccessors(currentState)
                next = random.choice(successors)
                currentState = next[0]
                individual.append([next[0], next[1]])
            score = self.getFitness(currentState)
            self.population.append([individual, score])

    def getActions(self, problem):
        '''
        The main iteration in Evolutionary algorithms.
        You can use getFitness, generateLegalActions, mutation, crossover and other function to evolve the population.
        :param problem:
        :return: the best individual in the population
        '''
        self.problem = problem
        self.population = None
        self.generateLegalActions()
        for i in range(0, self.T):
            offspringList = []
            for j in range(0, int(self.offspringNum / 2)):
                parent1, parent2 = self.selectParents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)
                score1 = self.getFitness(offspring1[self.actionDim][0])
                score2 = self.getFitness(offspring2[self.actionDim][0])
                offspringList.append([offspring1, score1])
                offspringList.append([offspring2, score2])
            self.population = sorted(self.population, key=lambda x: x[1])[:self.popSize - self.offspringNum]
            self.population = self.population + offspringList
        actions = []
        self.population = sorted(self.population, key=lambda x: x[1])
        for i in range(1, self.actionDim + 1):
            actions.append(self.population[0][0][i][1])
        if self.population[0][0][self.actionDim][0] in positionDict.keys():
            positionDict[self.population[0][0][self.actionDim][0]] += 1
        else:
            positionDict[self.population[0][0][self.actionDim][0]] = 1
        return actions

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.getActions(problem)  # Find a path

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):  # You may need to add some conditions for taking the action
            return self.actions[i]
        else:  # You may need to use getAction multiple times
            self.actionIndex = 0
            problem = self.searchType(state)
            self.actions = self.getActions(problem)
            if len(self.actions) > 0:
                action = self.actions[self.actionIndex]
                self.actionIndex += 1
            else:
                action = Directions.STOP
            print(self.callFitness)
            return action
