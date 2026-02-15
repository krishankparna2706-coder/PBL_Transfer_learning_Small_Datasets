# Transfer Learning for Small Datasets (MVP)

A runnable Python project that implements **Phase 1 (MVP)** of a Transfer Learning pipeline for image classification on small datasets. It uses a pre-trained ResNet18 as a **feature extractor** (frozen backbone + trainable classification head), basic data augmentation, and includes training, evaluation, CLI inference, and a simple REST API.

## Features

- **Image classification** on the hymenoptera_data dataset (ants vs bees) or your own folder-based dataset
- **Feature extraction**: ResNet18 backbone frozen, only a new classification head is trained
- **Data augmentation**: RandomResizedCrop(224), RandomHorizontalFlip, ImageNet normalization
- **Training and evaluation**: Accuracy and loss on validation set; classification report and confusion matrix
- **Model save/load**: Best model weights and class names saved for inference
- **CLI inference**: `python predict.py <image_path>` to get a prediction from the command line
- **REST API**: POST `/predict` with an image file to get a JSON prediction (FastAPI)

## Project Structure

```
.
├── config.py          # Paths, hyperparameters, constants
├── data_utils.py      # Dataset download, transforms, DataLoaders
├── model_utils.py     # ResNet18 feature extractor (frozen + new head)
├── main.py            # Training and evaluation entry point
├── predict.py         # Command-line inference
├── app.py             # FastAPI inference server
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── Dockerfile         # Optional: containerize the API
├── data/              # Created when dataset is downloaded
│   └── hymenoptera_data/
│       ├── train/
│       └── val/
├── best_model.pth     # Created after training
└── class_names.json   # Created after training (for API/CLI)
```

## Environment Setup

### 1. Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) and then install the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## How to Run

### Training

From the project root:

```bash
python main.py
```

This will:

1. Download the hymenoptera_data dataset (if not already present) into `data/`
2. Train the feature-extractor model for 15 epochs (configurable)
3. Save the best model to `best_model.pth` and class names to `class_names.json`
4. Print a classification report and confusion matrix on the validation set

**Options:**

- `--data-dir <path>` – Path to an existing `hymenoptera_data` directory (must contain `train/` and `val/` with class subfolders)
- `--save-path <path>` – Where to save the best model (default: `best_model.pth`)
- `--epochs N` – Number of epochs (default: 15)
- `--lr <float>` – Learning rate (default: 0.001)
- `--batch-size N` – Batch size (default from config)

Example:

```bash
python main.py --epochs 20 --lr 0.001
```

### Command-line inference

After training:

```bash
python predict.py path/to/your/image.jpg
```

Optional: `--model <path>`, `--class-names <path>` to override default paths.

### Start the inference API

After training:

```bash
python app.py
```

The API will listen on `http://0.0.0.0:5000`.

- **Docs**: http://localhost:5000/docs  
- **Health**: GET http://localhost:5000/health  
- **Predict**: POST http://localhost:5000/predict with form field `file` = image file  

Example with `curl`:

```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/predict
```

Example response:

```json
{
  "prediction": "ants",
  "confidence": 0.9523,
  "probabilities": {"ants": 0.9523, "bees": 0.0477}
}
```

If the model is not trained yet, the server can start but `/predict` will return 503 until you run `main.py`.

## Live demo (website)

The repo includes a **demo page** (`demo.html`) so visitors can upload an image and see a prediction. The "Launch Demo" button on the presentation links to it.

For the demo to work when the site is hosted on **GitHub Pages** (or any static host), the **inference API must be running somewhere** the browser can reach:

1. **Deploy the API** to a free tier host that runs Python (e.g. [Render](https://render.com), [Railway](https://railway.app), or [Hugging Face Spaces](https://huggingface.co/spaces)). Upload this project, set the start command to `python app.py` or `uvicorn app:app --host 0.0.0.0 --port $PORT`, and include the trained `best_model.pth` and `class_names.json` (e.g. by training in the cloud or uploading artifacts).
2. **On the demo page**, paste the deployed API URL (e.g. `https://your-app.onrender.com`) in the "Demo API URL" field and click **Save**.
3. Upload an image and click **Classify**. The page will send the image to your API and show the prediction.

**Testing locally:** Run `python app.py` on your machine, then open `demo.html` in the browser. If the demo is served from the same origin as the API (e.g. you open the file from the same machine where the API runs), you can leave the API URL blank or use `http://localhost:5000`.

## Docker (optional)

Build and run the API in a container:

```bash
docker build -t transfer-learning-mvp .
docker run -p 5000:5000 -v "%CD%\best_model.pth:/app/best_model.pth" -v "%CD%\class_names.json:/app/class_names.json" transfer-learning-mvp
```

On Linux/macOS use `$(pwd)` instead of `%CD%`. Ensure `best_model.pth` and `class_names.json` exist (run training first) or the container will start but predict will return 503 until you copy the files into the container or mount them.

## Success criteria (MVP)

- Validation accuracy **≥ 85%** on hymenoptera_data (achievable with default settings)
- Reproducible training and inference
- Model saving/loading and basic REST API for predictions

## References

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- Product Requirements Document and MVP Strategy (provided)
- Technical Execution and Deployment Steps (provided)
