# Plastic Rice Detection using Deep Learning

This project uses deep learning to detect plastic rice in both cooked and raw rice images.

## Features
- Detects plastic rice in both raw and cooked forms
- Uses transfer learning with a pre-trained CNN (MobileNetV2)
- Simple web interface for image upload and prediction (Streamlit)

## Project Structure
```
plastic_rice_detection/
├── data/                # Place your images here (raw/cooked, real/plastic)
├── model/               # Saved trained models
├── rice_classifier.py   # Model training script
├── app.py               # Streamlit web app
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup
1. Clone this repository or copy the folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Prepare your dataset:
   - Place images in `data/` folder, organized as:
     - `data/raw/real/`, `data/raw/plastic/`, `data/cooked/real/`, `data/cooked/plastic/`

## Training
Run the training script:
```
python rice_classifier.py
```

## Running the Web App
```
streamlit run app.py
```

## Notes
- You must provide your own dataset of rice images.
- The model and app are for educational purposes and may require further tuning for production use. 