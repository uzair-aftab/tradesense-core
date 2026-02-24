---
description: How to launch the TradeSense HFP trading dashboard
---
// turbo-all

## Launch HFP War Room

1. Activate the virtual environment:
```powershell
& "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\Activate.ps1"
```

2. Navigate to the project directory:
```powershell
Set-Location "C:\Python Virtual Environments\Personal Projects\Trading"
```

3. Launch the Streamlit dashboard:
```powershell
& "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\streamlit.exe" run src/ui/app.py --server.port 8501
```

4. The app will be available at http://localhost:8501

## Notes
- The virtual environment is at `C:\Python Virtual Environments\Personal Projects\TradeSense\`
- The project code is at `C:\Python Virtual Environments\Personal Projects\Trading\`
- Entry point: `src/ui/app.py`
- GPU: RTX 4070 (CUDA 11.8)
