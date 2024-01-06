# valueIterationAgents.py
# -----------------------
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

from learningAgents import ValueEstimationAgent
from copy import deepcopy
from random import random
from util import Counter


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.states = mdp.getStates()
        self.values = self.generate_initial_scores()

        """
            Implements Bellman's equation
        """
        for iteration in range(iterations):
            new_values = deepcopy(self.values)
            for state in self.states:
                if self.mdp.isTerminal(state) and state not in self.values:
                    continue
                if self.is_state_pre_terminal(state) and state not in self.values:
                    new_values[state] = mdp.getReward(state, None, None)
                    continue

                best_action, best_reward = self.get_best_action(state)
                new_values[state] = best_reward
            self.values = new_values

    def is_state_pre_terminal(self, state):
        for action in self.mdp.getPossibleActions(state):
            probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
            target_state, _ = sorted(probabilities, key=lambda prob: prob[1])[0]
            if self.mdp.isTerminal(target_state):
                return True
        return False

    def generate_initial_scores(self):
        values = Counter()
        for state in self.states:
            if self.mdp.isTerminal(state):
                continue
            values[state] = 0
            #values[state] = round(random(), 3)
        return values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for possible_state, probability in probabilities:
            if self.mdp.isTerminal(possible_state):
                continue
            possible_reward = self.getValue(possible_state)
            q_value += possible_reward * self.discount * probability
        return q_value

    def get_best_action(self, state):
        possible_actions = self.mdp.getPossibleActions(state)
        best_reward = float("-inf")
        best_action = None
        for action in possible_actions:
            reward = self.computeQValueFromValues(state, action)
            if reward >= best_reward:
                best_reward = reward
                best_action = action
        best_reward += self.mdp.getReward(state, best_action, None)
        return (best_action, best_reward)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        action, _ = self.get_best_action(state)
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
