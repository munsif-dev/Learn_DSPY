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
        # Break down complex reasoning into steps
        self.decompose = dspy.ChainOfThought("question -> subquestions")
        self.answer_sub = dspy.Predict("subquestion -> subanswer") 
        self.synthesize = dspy.ChainOfThought("question, subanswers -> final_answer")
    
    def forward(self, question):
        # Step 1: Break question into parts
        decomposition = self.decompose(question=question)
        
        # Step 2: Answer each part (simplified for demo)
        subquestions = decomposition.subquestions.split('\n')
        subanswers = []
        
        for subq in subquestions[:3]:  # Limit for demo
            if subq.strip():
                sub_pred = self.answer_sub(subquestion=subq.strip())
                subanswers.append(sub_pred.subanswer)
        
        # Step 3: Combine everything into final answer
        subanswers_text = '\n'.join(subanswers)
        final = self.synthesize(
            question=question, 
            subanswers=subanswers_text
        )
        
        return dspy.Prediction(
            subquestions=decomposition.subquestions,
            subanswers=subanswers_text,
            reasoning=final.rationale,
            answer=final.final_answer
        )

# Example usage demonstrating different module types
def demonstrate_modules():
    print("=== Basic Predict Module ===")
    basic = BasicPredictor()
    result1 = basic("What is machine learning?")
    print(f"Answer: {result1.answer}\n")
    
    print("=== Chain of Thought Module ===")
    reasoning = ReasoningPredictor()
    result2 = reasoning("Why is the sky blue?")
    print(f"Reasoning: {result2.reasoning}")
    print(f"Answer: {result2.answer}\n")
    
    print("=== Complex Multi-step Module ===")
    complex_qa = ComplexQA()
    result3 = complex_qa("How does climate change affect ocean ecosystems?")
    print(f"Subquestions: {result3.subquestions}")
    print(f"Subanswers: {result3.subanswers}")
    print(f"Final Answer: {result3.answer}")

# Each module type serves different purposes:
# - Predict: Direct input->output transformation
# - ChainOfThought: Adds explicit reasoning steps
# - ReAct: Can use tools and external actions
# - Custom modules: Combine multiple steps for complex workflows