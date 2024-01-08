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
from random import random  # noqa
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
                new_values[state] = float("-inf")
                sum_v = 0

                for action in self.mdp.getPossibleActions(state):
                    sum_v = self.computeQValueFromValues(state, action)

                    if sum_v > new_values[state]:
                        new_values[state] = sum_v

                if new_values[state] == float('-inf'):
                    new_values[state] = 0
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
        if self.mdp.isTerminal(state):
            return 0.0

        q_value = 0.0
        probabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        for possible_state, probability in probabilities:
            possible_reward = self.mdp.getReward(state, action, possible_state)
            possible_value = self.getValue(possible_state)
            q_value += probability * (possible_reward + self.discount * possible_value)
        return q_value

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

        best_value = float("-inf")
        best_action = None
        possible_actions = self.mdp.getPossibleActions(state)
        for action in possible_actions:
            value = self.computeQValueFromValues(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
