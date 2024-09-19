# SentiBot: Sentiment-Aware Llama 2 Chatbot

## Project Overview

SentiBot is an innovative chatbot that combines the power of Llama 2, a state-of-the-art language model, with sentiment analysis to provide context-aware and emotionally intelligent responses. This project was developed for the [Hackathon Name] and aims to showcase the integration of advanced NLP techniques in conversational AI.

## Features

1. **Sentiment Analysis Integration**: SentiBot analyzes the sentiment of user input to provide empathetic and context-appropriate responses.
2. **Llama 2 Integration**: Utilizes the Llama 2 model for generating high-quality, contextually relevant responses.
3. **Dynamic QA Dataset**: Maintains and updates a CSV-based question-answer dataset for quick responses to common queries.
4. **Gradio Web Interface**: Offers an intuitive and interactive web interface for easy interaction with the chatbot.

## Technical Details

- **Model**: NousResearch/Llama-2-7b-chat-hf
- **Sentiment Analysis**: Hugging Face Transformers pipeline
- **Frontend**: Gradio
- **Data Storage**: CSV file for QA pairs

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/sentibot.git
   cd sentibot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face API key as an environment variable:
   ```
   export HUGGINGFACE_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```
   python app.py
   ```

## Usage

1. Open the Gradio interface in your web browser.
2. Type your question or statement in the input box.
3. SentiBot will analyze the sentiment, generate a response, and display it in the output box.

## Code Structure

- `app.py`: Main application file containing the chatbot logic and Gradio interface.
- `qa_dataset.csv`: CSV file storing question-answer pairs.
- `requirements.txt`: List of Python dependencies.

## Future Enhancements

- Implement multi-turn conversations to maintain context over multiple interactions.
- Integrate more advanced sentiment analysis techniques for nuanced emotion detection.
- Expand the QA dataset with a wider range of topics and questions.

## Contributors

- [Your Name]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for providing the Transformers library and model hosting.
- The creators of Llama 2 for the powerful language model.
- Gradio for the easy-to-use web interface framework.

