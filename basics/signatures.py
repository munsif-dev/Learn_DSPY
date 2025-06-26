# First, let's install DSPy
# pip install dspy-ai

import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure DSPy with Gemini (you'll need to set up the appropriate LM)
# Note: You may need to use a different LM or create a custom Gemini wrapper for DSPy
# For now, I'll show the structure assuming you have a DSPy-compatible LM
lm = dspy.LM('gemini-2.5-flash-preview-04-17', api_key=GEMINI_API_KEY)
dspy.configure(lm=lm)
print(GEMINI_API_KEY)
print("DSPy configured successfully!")
print(f"Current language model: {dspy.settings.lm}")


##########################################################################
## Signatures in DSPy
##########################################################################

# Basic signature - like a function signature but for LM tasks
# Format: "input_field -> output_field"
basic_qa = dspy.Signature("question -> answer")

# More complex signature with multiple inputs
multi_input = dspy.Signature("context, question -> answer")

# You can also add descriptions to make your intent clearer
class DetailedQA(dspy.Signature):
    """Answer questions based on given context with detailed reasoning."""
    
    context = dspy.InputField(desc="Background information to base the answer on")
    question = dspy.InputField(desc="The question to be answered")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
    answer = dspy.OutputField(desc="Final concise answer")

def demonstrate_signatures():
    # Create a predictor using the signature
    qa_predictor = dspy.Predict(DetailedQA)
    
    # Use it to process a question
    context = "The capital of France is Paris. Paris is known for the Eiffel Tower."
    question = "What is the capital of France?"
    
    result = qa_predictor(context=context, question=question)
    
    print("Question:", question)
    print("Context:", context)
    print("Reasoning:", result.reasoning)
    print("Answer:", result.answer)
    
    return result


if __name__ == "__main__":
    result = demonstrate_signatures()
    print("Demonstration completed successfully!")
    print("Result:", result)
    print("Reasoning:", result.reasoning)
    print("Answer:", result.answer)