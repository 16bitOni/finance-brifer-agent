# Financial Assistant with Voice Interface

An intelligent financial assistant that provides portfolio analysis, market insights, and real-time stock information through a natural voice interface.

## Features

- **Voice Interface**: Speak to the assistant and get voice responses
- **Portfolio Analysis**: Get detailed insights about your investment portfolio
- **Real-time Market Data**: Access current stock prices and market trends
- **Earnings Analysis**: Track earnings surprises and company performance
- **News Integration**: Stay updated with relevant financial news
- **Risk Assessment**: Analyze portfolio risk exposure and concentration

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the root directory
- Add your API keys and credentials:
```
OPENROUTER_API_KEY=your_key_here
GOOGLE_APPLICATION_CREDENTIALS=path_to_credentials.json
```

5. Run the application:
```bash
streamlit run main.py
```

## Project Structure

```
├── agents/             # AI agents for different tasks
├── configs/            # Configuration files
├── services/           # External service integrations
├── state/             # State management
├── tools/             # Utility tools and functions
├── workflows/         # Business logic workflows
├── main.py            # Main application entry point
└── requirements.txt   # Project dependencies
```

## Technologies Used

- Streamlit for the web interface
- Google Cloud Text-to-Speech for voice output
- Google Cloud Speech-to-Text for voice input
- LangChain for AI/LLM integration
- Various financial data APIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 