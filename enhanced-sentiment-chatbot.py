import pandas as pd
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import gradio as gr

# Hugging Face Authentication
api_key = os.environ.get('HUGGINGFACE_API_KEY')
if not api_key:
    raise ValueError("No API key found. Please set the HUGGINGFACE_API_KEY environment variable.")

login(token=api_key)

# Initialize CSV file for QA dataset
csv_file = 'qa_dataset.csv'
if not os.path.exists(csv_file):
    qa_data = {
        'question': ["What is the name of Julius Magellan's dog?", "Who is Julius Magellan's dog?"],
        'answer': ["The name of Julius Magellan's dog is Sparky", "Julius Magellan's dog is called Sparky"]
    }
    qa_df = pd.DataFrame(qa_data)
    qa_df.to_csv(csv_file, index=False)
else:
    qa_df = pd.read_csv(csv_file)

# Initialize Llama 2 model and tokenizer
model_id = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.use_default_system_prompt = False

# Initialize Llama 2 pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    max_length=1024,
)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def answer_question(question):
    """
    Answers a question using the QA dataset or Llama 2 model.
    
    Args:
    question (str): The input question from the user.
    
    Returns:
    str: The answer to the question.
    """
    global qa_df
    answer = qa_df[qa_df['question'].str.lower() == question.lower()]['answer']
    
    if not answer.empty:
        return f"Answer from QA dataset: {answer.iloc[0]}"
    else:
        response = llama_pipeline(question, max_length=150, do_sample=True)[0]['generated_text']
        response = response.replace(f"Answer: {question}", "").strip()
        
        if not any(qa_df['question'].str.lower() == question.lower()):
            new_row = pd.DataFrame({'question': [question], 'answer': [response]})
            qa_df = pd.concat([qa_df, new_row], ignore_index=True)
            qa_df.to_csv(csv_file, index=False)
            return f"Answer from Llama 2: {response} \n(New QA pair added to the dataset.)"
        return f"Answer from Llama 2: {response}"

def analyze_sentiment(user_input):
    """
    Analyzes the sentiment of the user input.
    
    Args:
    user_input (str): The input text from the user.
    
    Returns:
    tuple: A tuple containing the sentiment label and score.
    """
    result = sentiment_analyzer(user_input)[0]
    label = result['label']
    score = result['score']
    return label, score

def sentiment_aware_response(question):
    """
    Generates a sentiment-aware response to the user's question.
    
    Args:
    question (str): The input question from the user.
    
    Returns:
    str: A sentiment-adjusted response to the question.
    """
    label, score = analyze_sentiment(question)
    
    if label == "NEGATIVE":
        sentiment_adjustment = "I sense some frustration. Let me try to help: "
    elif label == "POSITIVE":
        sentiment_adjustment = "I'm glad you're in a good mood! Here's what I found: "
    else:
        sentiment_adjustment = ""
    
    main_response = answer_question(question)
    full_response = f"{sentiment_adjustment}{main_response}"
    
    return full_response

# Gradio Interface
interface = gr.Interface(
    fn=sentiment_aware_response,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="SentiBot: Llama 2 Chatbot with Sentiment Analysis",
    description="Ask a question, and SentiBot will analyze the sentiment, adjust the response, and provide an answer using the QA dataset or Llama 2.",
    examples=[
        ["I'm having a great day! What's the capital of France?"],
        ["I'm feeling frustrated. Can you help me understand quantum physics?"],
        ["What's the weather like today?"]
    ],
    theme="huggingface"
)

# Launch the Gradio Interface
interface.launch()
