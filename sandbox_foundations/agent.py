"""Phase 1 Agent: simple prediction and learning loop."""

from environment import Action, ToyUniverseEnvironment


class SimplePredictiveModel:
    """Minimalist predictive brain mapping actions to success likelihood."""

    def __init__(self):
        # Key: action name, Value: estimated probability of success
        self.knowledge: dict[str, float] = {}

    def predict(self, action: Action) -> float:
        """Return the learned success probability for an action."""
        return self.knowledge.get(action.name, 0.5)

    def learn(self, action: Action, outcome_is_success: bool, learning_rate: float = 0.1):
        """Update the stored probability estimate based on the observed outcome."""
        prediction = self.predict(action)
        actual = 1.0 if outcome_is_success else 0.0
        error = actual - prediction
        new_estimate = prediction + learning_rate * error
        self.knowledge[action.name] = new_estimate
        print(f"🧠 Brain Update: Knowledge for {action.name} is now {new_estimate:.2f}")


class Agent:
    """Toy universe agent that predicts, acts, and learns."""

    def __init__(self):
        self.env = ToyUniverseEnvironment()
        self.brain = SimplePredictiveModel()

    def run_learning_loop(self, episodes: int = 10):
        """Iteratively act in the environment and update predictions."""
        print("🚀 Starting Learning Loop...")
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1}/{episodes} ---")
            action_to_take = Action.CHANGE_OPERATOR_ADD_TO_SUB

            predicted_success_prob = self.brain.predict(action_to_take)
            print(f"🤖 Agent Prediction: 'I think the chance of success is {predicted_success_prob:.2f}'")

            actual_outcome_is_success = self.env.step(action_to_take)
            self.brain.learn(action_to_take, actual_outcome_is_success)


def main():
    """Run the agent learning loop."""
    agent = Agent()
    agent.run_learning_loop()


if __name__ == "__main__":
    main()
