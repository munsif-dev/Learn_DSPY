import dspy

# Predict - the main class for making predictions
class BasicPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define what this module should do
        self.generate_answer = dspy.Predict("question -> answer")

        def forward(self, question):
            prediction = self.generate_answer(question=question)
            return dspy.Prediction(anwser=prediction.answer)

# Chainofthought - a module for reasoning through steps
class ChainOfThought(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define the chain of thought process
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

        def forward(self, context, question):
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(
                answer=prediction.answer, 
                reasoning=prediction.rationale
            )
        
# ReAct - a module for interactive question answering
# what ReAct does?
# it combines reasoning and action to answer questions
# it allows the model to think step by step and take actions based on the reasoning
# it is useful for complex tasks that require multiple steps to complete
# it can be used for tasks like question answering, planning, and decision making
# it is a powerful tool for building intelligent systems that can reason and act
# it is a module that can be used in DSPy to build intelligent systems that can reason and act
class ReAct(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define the ReAct process
        self.react = dspy.ReAct("question -> answer")

        def forward(self, question):
            prediction = self.react(question=question)
            return prediction
        
    

# Multi-Step pipeline - a module for handling complex tasks with multiple steps
class ComplexQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define the multi-step process
        self.decompose = dspy.ChainOfThought("question -> subquestions")
        self.answer_subquestions = dspy.Predict("subquestion -> subanswer")
        self.systhesize = dspy.ChainOfThought("question, subanswers -> final_answer")

        def forward(self, question):
        