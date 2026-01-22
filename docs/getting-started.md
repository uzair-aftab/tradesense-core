# Getting Started

This guide will help you set up TradeSense on your local machine.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.12** installed ([download](https://www.python.org/downloads/))
- **Git** installed ([download](https://git-scm.com/downloads))
- **NVIDIA GPU** with CUDA 12.4 support (optional but recommended)

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/uzair-aftab/tradesense.git
cd tradesense
```

### 2. Create Virtual Environment

=== "Windows (PowerShell)"
    ```powershell
    py -3.12 -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

=== "Linux/macOS"
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```

### 3. Install PyTorch with CUDA

=== "With GPU (Recommended)"
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

=== "CPU Only"
    ```bash
    pip install torch torchvision torchaudio
    ```

### 4. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

### 5. Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit .env with your API keys (optional but recommended)
```

## API Keys (Optional)

TradeSense works without API keys, but having them enables more features:

| Service | Free Tier | Used For |
|---------|-----------|----------|
| [Finnhub](https://finnhub.io/register) | 60 calls/min | News, sentiment, fundamentals |
| [Reddit API](https://www.reddit.com/prefs/apps) | 60 req/min | WSB sentiment analysis |
| [Marketaux](https://www.marketaux.com/register) | 100 calls/day | Backup news source |

## Verify Installation

```python
# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test FinBERT model loading
python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('ProsusAI/finbert'); print('FinBERT loaded!')"
```

## Running TradeSense

### Start the Dashboard

```bash
streamlit run app/main.py
```

The dashboard will open at `http://localhost:8501`

### Run a Quick Scan

```python
from src.signals.generator import SignalGenerator

scanner = SignalGenerator()
signals = scanner.scan(['AAPL', 'MSFT', 'NVDA'])
print(signals)
```

## Next Steps

- [Configuration](configuration.md) - Customize thresholds and watchlists
- [Modules Overview](modules/data-layer.md) - Understand the architecture
- [API Reference](api/reference.md) - Detailed API documentation

## Troubleshooting

!!! tip "Common Issues"
    **CUDA not detected**: Ensure you have the correct NVIDIA drivers installed for CUDA 12.4
    
    **Import errors**: Make sure your virtual environment is activated
    
    **API rate limits**: Consider adding API keys for higher limits
