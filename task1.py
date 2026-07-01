"""
============================================================
  MOVIE GENRE CLASSIFICATION — Complete ML Pipeline
  Dataset: IMDB Genre Classification (Kaggle - hijest)
============================================================
"""

# ── 1. IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

warnings.filterwarnings('ignore')

print("=" * 60)
print("  MOVIE GENRE CLASSIFICATION — ML Pipeline")
print("=" * 60)


# ── 2. LOAD REAL KAGGLE DATASET ─────────────────────────────
# ⚠️  Change YourName to your PC username!
# Example: C:\Users\Lacshini\Downloads\train_data.txt

df = pd.read_csv(
    r'C:\Users\panda\Downloads\archive\Genre Classification Dataset\train_data.txt',
    sep=' ::: ',
    engine='python',
    names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'],
    index_col=False
)
# Remove header row if it got read as data
df = df[df['ID'] != 'ID'].reset_index(drop=True)

print(f"\n✅  Dataset loaded: {len(df)} movies")
print(f"    Genres found: {sorted(df['GENRE'].unique())}")


# ── 3. TEXT PREPROCESSING ────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_desc']  = df['DESCRIPTION'].apply(clean_text)
df['clean_title'] = df['TITLE'].apply(clean_text)
df['features']    = df['clean_title'] + " " + df['clean_desc']

print("✅  Text cleaning done.")


# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────
X = df['features']
y = df['GENRE'].str.strip()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅  Split: {len(X_train)} train | {len(X_test)} test")


# ── 5. THREE CLASSIFIERS ────────────────────────────────────
tfidf_params = dict(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words='english'
)

models = {
    "Naive Bayes": Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   MultinomialNB(alpha=0.5))
    ]),
    "Logistic Regression": Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    "Linear SVM": Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   LinearSVC(C=1.0, max_iter=2000, random_state=42))
    ])
}

results = {}
print("\n── Training Models ──────────────────────────────────────")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'model': pipe, 'pred': y_pred, 'acc': acc, 'f1': f1}
    print(f"  {name:<22}  Accuracy={acc:.2%}  F1={f1:.2%}")


# ── 6. BEST MODEL ───────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best      = results[best_name]
print(f"\n🏆  Best Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, best['pred']))


# ── 7. VISUALISATION ────────────────────────────────────────
genres  = sorted(df['GENRE'].str.strip().unique())
palette = {
    'action':'#E74C3C', 'drama':'#3498DB',
    'horror':'#8E44AD', 'romance':'#E91E63',
    'sci-fi':'#00BCD4', 'comedy':'#F39C12',
    'thriller':'#27AE60', 'documentary':'#E67E22'
}

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0D1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
title_kw = dict(color='white', fontsize=12, fontweight='bold', pad=10)

# A: Genre Distribution
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor('#161B22')
counts = df['GENRE'].str.strip().value_counts()
cols   = [palette.get(g, '#95A5A6') for g in counts.index]
bars   = ax0.bar(counts.index, counts.values, color=cols, edgecolor='none', width=0.6)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', color='white', fontsize=9)
ax0.set_title("Genre Distribution", **title_kw)
ax0.set_xlabel("Genre", color='#8B949E')
ax0.set_ylabel("Count", color='#8B949E')
ax0.tick_params(colors='#8B949E', rotation=30)
ax0.spines[:].set_visible(False)

# B: Model Comparison
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor('#161B22')
mnames = list(results.keys())
accs   = [results[m]['acc'] for m in mnames]
f1s    = [results[m]['f1']  for m in mnames]
x, w   = np.arange(len(mnames)), 0.35
b1 = ax1.bar(x - w/2, accs, w, label='Accuracy', color='#3498DB', edgecolor='none')
b2 = ax1.bar(x + w/2, f1s,  w, label='F1-Score',  color='#2ECC71', edgecolor='none')
for bars in [b1, b2]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.0%}", ha='center', color='white', fontsize=8)
ax1.set_title("Model Comparison", **title_kw)
ax1.set_xticks(x)
ax1.set_xticklabels(['Naive\nBayes','Logistic\nRegr.','Linear\nSVM'], color='#8B949E')
ax1.set_ylim(0, 1.12)
ax1.tick_params(colors='#8B949E')
ax1.spines[:].set_visible(False)
ax1.legend(facecolor='#161B22', labelcolor='white', fontsize=9)

# C: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#161B22')
cm = confusion_matrix(y_test, best['pred'], labels=genres)
ax2.imshow(cm, cmap='Blues')
ax2.set_xticks(range(len(genres)))
ax2.set_yticks(range(len(genres)))
ax2.set_xticklabels(genres, rotation=30, ha='right', color='#8B949E', fontsize=8)
ax2.set_yticklabels(genres, color='#8B949E', fontsize=8)
for i in range(len(genres)):
    for j in range(len(genres)):
        ax2.text(j, i, str(cm[i,j]), ha='center', va='center',
                 color='white' if cm[i,j] > cm.max()/2 else '#8B949E', fontsize=10)
ax2.set_title(f"Confusion Matrix ({best_name})", **title_kw)
ax2.set_xlabel("Predicted", color='#8B949E')
ax2.set_ylabel("Actual",    color='#8B949E')

# D: Per-genre F1
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#161B22')
report  = classification_report(y_test, best['pred'], output_dict=True)
gf1     = {g: report[g]['f1-score'] for g in genres if g in report}
gcols   = [palette.get(g,'#95A5A6') for g in gf1]
bars3   = ax3.barh(list(gf1.keys()), list(gf1.values()), color=gcols, edgecolor='none')
for bar, val in zip(bars3, gf1.values()):
    ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va='center', color='white', fontsize=9)
ax3.set_xlim(0, 1.2)
ax3.set_title("Per-Genre F1 Score", **title_kw)
ax3.tick_params(colors='#8B949E')
ax3.spines[:].set_visible(False)

# E: Top Keywords per Genre
ax4 = fig.add_subplot(gs[1, 1:])
ax4.set_facecolor('#161B22')
ax4.axis('off')
feat_names = best['model'].named_steps['tfidf'].get_feature_names_out()
clf        = best['model'].named_steps['clf']
top_words  = {}
if hasattr(clf, 'coef_'):
    classes = clf.classes_
    for i, g in enumerate(classes):
        idx = np.argsort(clf.coef_[i])[-6:][::-1]
        top_words[g] = [feat_names[j] for j in idx]
else:
    for i, g in enumerate(clf.classes_):
        idx = np.argsort(clf.feature_log_prob_[i])[-6:][::-1]
        top_words[g] = [feat_names[j] for j in idx]

col_labels = list(top_words.keys())[:5]
cell_text, cell_colors = [], []
for rank in range(6):
    row, rcol = [], []
    for g in col_labels:
        words = top_words.get(g, [])
        row.append(words[rank] if rank < len(words) else '')
        base = palette.get(g,'#95A5A6')
        r,gb,b = int(base[1:3],16), int(base[3:5],16), int(base[5:7],16)
        a = 0.15 + 0.05*(5-rank)
        rcol.append((r/255*a+0.08, gb/255*a+0.08, b/255*a+0.08, 1))
    cell_text.append(row)
    cell_colors.append(rcol)

tbl = ax4.table(cellText=cell_text, rowLabels=[f"#{i+1}" for i in range(6)],
                colLabels=[g.upper() for g in col_labels],
                cellColours=cell_colors, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.4, 1.8)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#30363D')
    cell.set_text_props(color='white')
    if r == 0 or c == -1:
        cell.set_facecolor('#21262D')
        cell.set_text_props(color='white', fontweight='bold')
ax4.set_title("Top TF-IDF Keywords per Genre", **title_kw)

fig.suptitle("🎬  Movie Genre Classification — ML Pipeline",
             color='white', fontsize=16, fontweight='bold', y=0.98)

# Save to Desktop
import os
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, "movie_genre_result.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.show()
print(f"\n✅  Chart saved to Desktop: movie_genre_result.png")


# ── 8. TEST WITH NEW PLOTS ──────────────────────────────────
print("\n── Test Predictions ─────────────────────────────────────")
test_plots = [
    "A soldier fights enemies in a war zone battle",
    "Two people fall in love at a coffee shop",
    "A ghost haunts a family in their new house",
    "Astronauts travel to another galaxy in space",
    "A man struggles with poverty and family problems"
]
preds = best['model'].predict(test_plots)
for plot, pred in zip(test_plots, preds):
    print(f"  {plot[:50]:<52} → {pred.upper()}")

print("\n" + "="*60)
print(f"  ✅  Best Model : {best_name}")
print(f"  ✅  Accuracy   : {best['acc']:.2%}")
print(f"  ✅  F1 Score   : {best['f1']:.2%}")
print("="*60)
