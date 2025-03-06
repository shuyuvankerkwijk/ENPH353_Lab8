import random
import pickle

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        filename = filename if filename.endswith(".pickle") else filename + ".pickle"
        with open(filename, "rb") as f:
            self.q = pickle.load(f)
        print("Loaded file: {}".format(filename))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        filename = filename if filename.endswith(".pickle") else filename + ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(self.q, f)
        print("Wrote to file: {}".format(filename))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        if random.random() <= self.epsilon: # choose random action
            action = random.choice(self.actions)
            max_q =  0.0
        else: # choose action with highest Q value for our current state
            q_values = {action: self.getQ(state, action) for action in self.actions}
            max_q = max(q_values.values())
            best_actions = [action for action,q in q_values.items() if q==max_q]
            action = random.choice(best_actions)

        return (action, max_q) if return_q else action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        s1_q = self.getQ(state1, action1)
        s2_qs = [self.getQ(state2, action2) for action2 in self.actions]
        max_s2_q = max(s2_qs)

        # update using Bellman update function
        updated_s1_q = s1_q + self.alpha*(reward + self.gamma*max_s2_q - s1_q)
        self.q[(state1, action1)] = updated_s1_q
