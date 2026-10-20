# Student Performance Analysis System

A Streamlit dashboard that turns a CSV of student marks into a cleaned dataset, grades, ranks,
charts and downloadable reports. Built for a college PBL activity (Application / Software Development).

**Stack:** Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, ReportLab

## Features

| Area | What it does |
|---|---|
| CSV input | Upload your own `.csv` or use the built-in sample dataset. A downloadable template is provided. |
| Cleaning and validation | Removes empty rows, exact duplicates and repeated Student IDs; flags text in mark columns; rejects marks outside 0-100; handles missing marks (median / zero / drop). Every action appears in a cleaning log. |
| Statistics | Mean, median, standard deviation, pass percentage, class average and subject-wise topper. |
| Grades | Total, percentage, grade (A+, A, B, C, D, F) and pass/fail. |
| Charts | Subject averages, pass/fail pie, histogram, boxplot, grade distribution, correlation heatmap (Matplotlib + Seaborn) plus native Streamlit charts. |
| Ranking | Rank, percentile, search by name/ID, subject-wise ranking, grade/result filters, student profile versus class average. |
| Export | Processed results as CSV, formatted multi-page PDF (ReportLab), Markdown summary. |

## Project structure

```
student_performance_analysis/
├── app.py                 # Streamlit dashboard + all logic
├── requirements.txt       # Dependencies
├── sample_students.csv    # 20 students x 6 subjects (with a few deliberate data errors)
├── README.md
└── .streamlit/config.toml # Theme and server settings
```

## Run locally

```bash
# 1. Extract the zip and open a terminal inside the folder
cd student_performance_analysis

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Using your own CSV

One row per student, one column per subject, marks out of 100:

```
Student_ID,Name,Maths,Science,English
S001,Asha Rao,78,65,72
S002,Vikram Singh,56,48,60
```

- The app guesses the name, ID and subject columns. Fix them under **Column mapping** in the sidebar if needed.
- The Student ID column is optional.
- Change the **pass mark** (default 40) and the **missing marks** strategy in the sidebar.

## Rules used

- **Percentage** = total marks / (number of subjects x 100) x 100
- **Pass** = every subject is at or above the pass mark. Otherwise **Fail**.
- **Grade**: A+ 90 and above, A 80-89.99, B 70-79.99, C 60-69.99, D 40-59.99, F below 40.
  A student who fails any subject is graded F.
- **Rank** is by total marks; tied students share a rank. **Percentile** is the share of students at or below that score.
- **Std deviation** uses the sample formula (n - 1).

## Deploy on Streamlit Community Cloud

1. Create a free GitHub account and a **new public repository** (for example `student-performance-analysis`).
2. Upload the **contents** of this folder so that `app.py` and `requirements.txt` sit at the root of the repository
   (include `sample_students.csv` and the `.streamlit` folder).
   ```bash
   git init
   git add .
   git commit -m "Student Performance Analysis System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/student-performance-analysis.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** (or **New app**), choose **Deploy a public app from GitHub**.
5. Select your repository, branch `main`, and set **Main file path** to `app.py`.
6. Open **Advanced settings** and choose Python 3.11 or 3.12 (3.10+ works).
7. Click **Deploy**. The first build takes a few minutes. You get a public `*.streamlit.app` link
   that works on phones and desktops.

Any later `git push` redeploys the app automatically.

## Troubleshooting

- **ModuleNotFoundError on deploy**: check the package is listed in `requirements.txt` and that the file is at the repository root.
- **Sample dataset not found**: `sample_students.csv` must sit in the same folder as `app.py`.
- **"No usable records left"**: open Column mapping and make sure the subject columns are numeric marks.
- **Upload fails**: save the file as CSV (UTF-8) and check it has a header row.
