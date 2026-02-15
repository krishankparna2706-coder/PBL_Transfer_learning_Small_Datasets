# Deploy the API (Option A) so the Live Demo works

Follow these steps to deploy the inference API to **Render** so anyone visiting your GitHub Pages site can use the "Launch Demo" and get real predictions.

---

## 1. Train the model and add files to the repo

The API needs `best_model.pth` and `class_names.json` to run. Train locally, then add them to the repo:

```powershell
cd "c:\Users\krish\OneDrive\Desktop\PBL_Tranfer__learning_Datasets"
python main.py
```

After training, force-add the model files (they are in `.gitignore` by default) and push:

```powershell
git add -f best_model.pth class_names.json
git commit -m "Add trained model for API deployment"
git push
```

---

## 2. Deploy on Render

1. Go to **[Render](https://render.com)** and sign in (GitHub login is fine).
2. Click **New** → **Web Service**.
3. Connect your repository: **krishankparna2706-coder/PBL_Transfer_learning_Small_Datasets**.
4. Configure the service:
   - **Name:** `pbl-transfer-learning-api` (so the URL matches the one in `api-config.js`).
   - **Region:** Choose the closest to you.
   - **Branch:** `main`.
   - **Runtime:** Python 3.
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. **Plan:** Free.
6. Click **Create Web Service**. Render will build and deploy. The first deploy may take a few minutes (and the free tier may spin down after inactivity).

Your API URL will be: **https://pbl-transfer-learning-api.onrender.com**

(If you chose a different name, update `api-config.js` with your URL and push.)

---

## 3. API URL in the demo

The repo already has **`api-config.js`** with:

```js
window.DEMO_API_BASE = "https://pbl-transfer-learning-api.onrender.com";
```

The demo page (`demo.html`) loads this file and uses this URL by default, so visitors do not need to paste anything. If your Render service name is different, edit `api-config.js`, set `window.DEMO_API_BASE` to your URL (e.g. `https://your-service-name.onrender.com`), then commit and push.

---

## 4. Check that it works

1. Open your GitHub Pages site → **Launch Demo**.
2. The "Demo API URL" field should be pre-filled with the Render URL (from `api-config.js`). If not, paste it and click **Save**.
3. Upload an image of an ant or bee and click **Classify**. You should see the prediction and confidence.

**Note:** On the free tier, Render may put the service to sleep after ~15 minutes of no traffic. The first request after that can take 30–60 seconds to wake up; later requests are fast.
