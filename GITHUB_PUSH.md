# Push This Project to GitHub

Follow these steps to upload the project (including `index.html`) to your repository:

**Repository:** https://github.com/krishankparna2706-coder/PBL_Transfer_learning_Small_Datasets

---

## 1. Install Git (if not installed)

- Download: https://git-scm.com/download/win  
- Install and ensure "Git from the command line" is enabled so `git` works in PowerShell/CMD.

---

## 2. Open terminal in the project folder

```powershell
cd "c:\Users\krish\OneDrive\Desktop\PBL_Tranfer__learning_Datasets"
```

---

## 3. Initialize Git and add files

```powershell
git init
git add config.py data_utils.py model_utils.py main.py predict.py app.py requirements.txt README.md Dockerfile .gitignore index.html GITHUB_PUSH.md
git status
```

(If you want to add everything except what’s in `.gitignore`: `git add .`)

---

## 4. First commit

```powershell
git commit -m "Transfer Learning MVP: training, CLI, API, and PBL presentation index.html"
```

---

## 5. Connect to your GitHub repo and push

Replace `YOUR_USERNAME` with your GitHub username if you use HTTPS; or use SSH if you have keys set up.

**Option A – HTTPS (will ask for GitHub username and password/token):**

```powershell
git remote add origin https://github.com/krishankparna2706-coder/PBL_Transfer_learning_Small_Datasets.git
git branch -M main
git push -u origin main
```

**Option B – If the repo already has content (e.g. README):**

```powershell
git remote add origin https://github.com/krishankparna2706-coder/PBL_Transfer_learning_Small_Datasets.git
git branch -M main
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## 6. Enable GitHub Pages (to host index.html)

1. Open: https://github.com/krishankparna2706-coder/PBL_Transfer_learning_Small_Datasets  
2. **Settings** → **Pages**  
3. Under **Source**, choose **Deploy from a branch**  
4. Branch: **main**, folder: **/ (root)**  
5. Save.

Your presentation will be available at:

**https://krishankparna2706-coder.github.io/PBL_Transfer_learning_Small_Datasets/**

---

## 7. Update placeholders in index.html

Edit `index.html` and replace:

- `[YOUR_REG_NO]` – your registration number  
- `[Guide Name]` – project guide name  
- `[Student Name]` and `[Reg No]` – your name and reg no  

Then commit and push again:

```powershell
git add index.html
git commit -m "Update PBL presentation with team details"
git push
```
