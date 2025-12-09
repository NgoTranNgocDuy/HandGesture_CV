### Rule-Based Detection
```powershell
pip install -r requirements.txt
python app.py
```
Open browser: `http://localhost:5000`

---

### Deep Learning Model (Recommended)
```powershell
pip install -r requirements_dl.txt
python dataset/data_collector.py  # Collect dataset (optional)
python train_model.py              # Train model (optional)
python app_with_dl.py              # Run application
```
Open browser: `http://localhost:5000`