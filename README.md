<div align="center">

# ML Analysis of Video Games

**What makes a video game successful? Exploring ratings, sales and game attributes with machine learning.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/UleleBulele/ML-Analysis-of-Video-Games/blob/main/Report_Project_Group_B06C_D.ipynb)

</div>

---

## Overview

The video game industry is fast-moving and highly competitive, with thousands of titles released across many platforms. Understanding what drives a game's commercial success is valuable to developers, publishers and marketers alike.

This **BUSAN 302 group project** analyses **6,250 video game releases** to answer one guiding question:

> **How can we gain insights into what makes a video game successful using ratings, sales and game attributes?**

We clean and explore the data, then apply three techniques that each look at success from a different angle:

| Technique | Purpose |
|---|---|
| **Linear regression** | Predict a game's global sales |
| **Logistic regression** | Classify whether a game is "popular" (1M+ sales) |
| **K-Means clustering** | Discover segments of games by reviews and sales |

## Key Findings

- **Predicting sales is hard.** Linear regression using review scores, genre, platform and release year explains only **13.5%** of the variance in global sales (R² = 0.135).
- **Review scores alone are weak predictors.** Critic score has a 0.24 correlation with global sales, and user score only 0.095.
- **Classification is better at recall than precision.** The logistic regression catches **75%** of popular games (1M+ sales), but only **40%** of the games it flags as popular actually are.
- **Clustering was the most useful model.** K-Means revealed three meaningful segments (underperformers, niche titles and top performers), which give more actionable insight than the predictive models.
- **Context matters.** Marketing, franchise strength, platform and release timing are likely to matter far more than review scores, but are absent from the data.

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Business Implications](#business-implications)
- [Limitations and Ethics](#limitations-and-ethics)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Team](#team)
- [References](#references)

## Dataset

`VideoGames.csv` contains **6,250 entries** of video games released between **1985 and 2016** across various platforms.

[Download Data File](VideoGames.csv)

| Variable | Description |
|---|---|
| `Name` | Title of the video game |
| `Platform` | Gaming system (e.g. PS2, Xbox, PC) |
| `Year_of_Release` | Release year |
| `Genre` | Game genre (e.g. Action, Sports, Shooter) |
| `NA_Sales` | Sales in North America (millions) |
| `EU_Sales` | Sales in Europe (millions) |
| `JP_Sales` | Sales in Japan (millions) |
| `Other_Sales` | Sales in the rest of the world (millions) |
| `Global_Sales` | Total worldwide sales (millions) |
| `Critic_Score` | Professional review rating (0 to 100) |
| `User_Score` | Player review rating (0 to 10) |

Most games sell relatively few copies, while a handful of hits skew the averages: the median game sells 0.29M copies, while the best seller reaches 82.53M. North America leads regional sales, and releases are concentrated between 2000 and 2011. Action, Sports and Shooter are the three most common genres out of 12.

<p align="center">
  <img width="630" height="470" alt="games_by_genre" src="https://github.com/user-attachments/assets/9e77c2bf-5803-43dd-9ff4-798fc95c331e" />
</p>

## Methodology

### 1. Data cleaning

Missing values appeared in all seven numeric columns. Rows with any missing value were dropped, since the model features and target must be complete.

| Step | Rows |
|---|---|
| Raw dataset | 6,250 |
| Rows removed (missing values) | 108 (1.7%) |
| **Clean dataset** | **6,142** |

### 2. Exploratory analysis

A correlation heatmap and pairplot were used to examine how review scores relate to sales. Regional sales are almost perfectly correlated with `Global_Sales` (0.62 to 0.96), because global sales is their sum. Critic and user scores agree moderately (0.58).

<p align="center">
  <img width="743" height="590" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/2c354642-ded8-4c17-9dfc-d8463ebfdb2f" />
</p>

<details>
<summary>View pairplot with regression lines</summary>
<br>
<p align="center">
  <img width="986" height="1023" alt="pairplot_regression" src="https://github.com/user-attachments/assets/3e0763b8-59b6-41de-8d5f-ccc28d2933f0" />
</p>
</details>

### 3. Preprocessing and feature engineering

- **Data leakage prevention:** `NA_Sales`, `EU_Sales`, `JP_Sales` and `Other_Sales` are excluded from the models, since they sum to the target.
- **One-hot encoding** of `Genre` and `Platform` (first category dropped), used in the regression models.
- **Standardisation** (`StandardScaler`) of `Critic_Score`, `User_Score` and `Year_of_Release`, so features on very different scales contribute equally.
- `Name` is dropped as an identifier.

### 4. Models

| Model | Task | Features | Setup |
|---|---|---|---|
| **Linear Regression** | Predict `Global_Sales` | Review scores, release year, one-hot genre and platform | 80/20 train/test split, `random_state=42` |
| **Logistic Regression** | Classify popular (`Global_Sales` ≥ 1M) vs not | Same as above | 80/20 split, balanced class weights |
| **K-Means (k = 3)** | Segment games | `Critic_Score`, `User_Score`, `Global_Sales` | Standardised features, `random_state=42` |

## Results

### Linear regression: predicting global sales

| Metric | Value |
|---|---|
| R² | **0.135** |
| RMSE | **1.447** (millions of units) |

Review scores, genre, platform and release year together explain only about 13.5% of the variation in sales, so the linear model has weak predictive power.

### Logistic regression: predicting popularity (1M+ sales)

Evaluated on a test set of 1,229 games.

| Metric | Not popular | Popular |
|---|---|---|
| Precision | 0.92 | 0.40 |
| Recall | 0.72 | 0.75 |
| F1-score | 0.81 | 0.52 |

**Overall accuracy: 0.73.**

| | Predicted not popular | Predicted popular |
|---|---|---|
| **Actually not popular** | 715 | 272 |
| **Actually popular** | 60 | 182 |

The model finds most popular games (high recall), but the low precision means it also labels many non-popular games as popular (272 false positives against 182 true positives). It is a useful screening tool rather than a reliable classifier.

### K-Means clustering: game segments

Silhouette score for three clusters: **0.298**, which indicates overlapping but distinguishable groups. The clusters should be read as general trends rather than precise categories.

| Cluster | Avg. critic score | Avg. user score | Avg. global sales (M) | Segment |
|---|---|---|---|---|
| 0 | 51.0 | 4.8 | 0.34 | **Underperformers:** low scores, very low sales |
| 1 | 66.8 | 7.2 | 0.38 | **Niche titles:** moderate scores, very low sales (possibly low-visibility games) |
| 2 | 82.6 | 8.1 | 1.52 | **Top performers:** high scores, much higher sales |

<p align="center">
  <img width="677" height="547" alt="kmeans_clusters" src="https://github.com/user-attachments/assets/3e23c4c1-a2b7-4d8e-9a45-2bfc80c8a181" />
</p>

Segmenting games this way surfaces patterns a single prediction cannot, such as highly rated games that sell poorly, or modestly rated games that sell well.

## Business Implications

- **Segment rather than predict.** Clustering gave more actionable insight than the regression models, and gives developers and publishers a framework for targeting different market segments.
- **Strong reviews with weak sales** may signal marketing or platform misalignment.
- **Modest reviews with high sales** suggest the influence of franchise power or strategic release timing.
- **Platform choice matters** when assessing success beyond a game's home market (Babb et al., 2013).
- **Engagement is increasingly important.** Active players and playtime are becoming critical success metrics as digital distribution and subscription models grow (Van Crombrugge & Stremersch, 2025).

## Limitations and Ethics

### Limitations

- **Narrow feature set.** Platform exclusivity, franchise branding, marketing budget and seasonal release timing are known to influence success but are not in the dataset. Success in the industry is complex and driven by factors beyond scores and sales (Marchand & Hennig-Thurau, 2013).
- **K-Means assumptions.** The algorithm assumes similarly sized, roughly spherical clusters, which real game performance rarely follows, especially for mid-ranked games that blend high- and low-scoring traits.
- **Weak linear fit.** Sales are heavily skewed, so a log transform or tree-based models (e.g. random forest, gradient boosting) may perform better.
- **Preprocessing before the split.** The scaler is fitted on the full dataset before the train/test split. A scikit-learn `Pipeline` would avoid fitting on test data.
- **Choice of k.** k = 3 was chosen by experimenting with different cluster counts; adding an elbow or silhouette comparison across k to the notebook would make this reproducible.
- **Missing values were dropped** rather than imputed (1.7% of rows).

### Ethical considerations

- **Review manipulation.** Review scores are subjective and can be distorted by coordinated review bombing, which may disadvantage niche or minority-supported games (Cantone et al., 2024).
- **Regional and format bias.** The data favours historically popular titles and franchises in Western and Japanese markets, and under-represents mobile games, indie games and non-English markets such as China, Southeast Asia and Latin America. Conclusions may not generalise to today's global gaming landscape.
- **Exploratory use only.** Clustering is best used for trend detection and strategy direction, not as a substitute for human judgement.

## Getting Started

### Option 1: Google Colab

Click the **Open in Colab** badge at the top of this page. Upload `VideoGames.csv` to your Google Drive at `MyDrive/VideoGames.csv`, as the notebook reads it from there.

### Option 2: Run locally

```bash
# Clone the repository
git clone https://github.com/UleleBulele/ML-Analysis-of-Video-Games.git
cd ML-Analysis-of-Video-Games

# Install dependencies (scikit-learn 1.2 or later is required)
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Report_Project_Group_B06C_D.ipynb
```

Before running locally, make two changes in the first cells of the notebook:

1. Remove the Google Drive mount (`from google.colab import drive` and `drive.mount(...)`).
2. Change the data path to your local copy: `pd.read_csv('VideoGames.csv')`.

## Project Structure

```
ML-Analysis-of-Video-Games/
├── Report_Project_Group_B06C_D.ipynb   # Full analysis and modelling notebook
├── docs/                               # Final report and presentation slides
├── images/                             # Charts used in this README
├── requirements.txt                    # Python dependencies
└── README.md
```

## Team

BUSAN 302 Group Project (Group B06C-D).

| Member | Contributions |
|---|---|
| **Sri Kadali** | Slide content, report support, dataset selection and project question |
| **Joshua Walton** | Executive summary, introduction, descriptive statistics, model results, Google Colab coding |
| **Lily Liang** | Report editing and structure, data preprocessing, conclusion |
| **Jeevesh Concisom** | Data preprocessing, report editing and improvement |

## References

- Babb, J., Terry, N., & Dana, K. (2013). The impact of platform on global video game sales. *International Business & Economics Research Journal (IBER)*, 12(10), 1273. https://doi.org/10.19030/iber.v12i10.8136
- Cantone, G. G., Tomaselli, V., & Mazzeo, V. (2024). Review bombing: ideology-driven polarisation in online ratings: The case study of The Last of Us (part II). *Quality & Quantity*. https://doi.org/10.1007/s11135-024-01981-z
- Marchand, A., & Hennig-Thurau, T. (2013). Value creation in the video game industry: industry economics, consumer benefits, and research opportunities. *Journal of Interactive Marketing*, 27(3), 141-157. https://doi.org/10.1016/j.intmar.2013.05.001
- Van Crombrugge, M., & Stremersch, S. (2025). Engagement in platform markets: A (video) game changer? *Journal of the Academy of Marketing Science*. https://doi.org/10.1007/s11747-025-01089-2
