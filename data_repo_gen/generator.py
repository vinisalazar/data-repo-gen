
import csv, random
from pathlib import Path
import nbformat as nbf
import pandas as pd

def _write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _infer_columns(csv_path: Path):
    df = pd.read_csv(csv_path)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return df, num_cols, cat_cols

def _make_messy_csv(df, out_csv: Path, tweak_schema=True, inject_na=True):
    rows = [df.columns.tolist()] + df.astype(str).values.tolist()
    if inject_na:
        for r in rows[1: min(6, len(rows))]:
            for j in range(len(r)):
                if random.random() < 0.1:
                    r[j] = "NA"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

def _make_nb_analysis(dataset_name: str, target: str, task: str, out_ipynb: Path):
    nb = nbf.v4.new_notebook(); cells=[]
    cells += [nbf.v4.new_markdown_cell(
        f"# {dataset_name.title()} Analysis (work in progress)\n\n"
        "_Intentionally messy notebook for educational use._")]
    cells += [nbf.v4.new_code_cell(
        "df = pd.read_csv('~/Downloads/DATA.csv')  # TODO: move into repo\n"
        "df.head()")]
    cells += [nbf.v4.new_code_cell(
        "# Premature filtering & dropping\n"
        "df = df.dropna()\n"
        "len(df)")]
    cells += [nbf.v4.new_code_cell(
        "sns.histplot(df.iloc[:,0])\n"
        "plt.title('First column hist')")]
    cells += [nbf.v4.new_code_cell(
        "import os, numpy as np, pandas as pd\n"
        "import matplotlib.pyplot as plt, seaborn as sns\n"
        "from sklearn.model_selection import train_test_split\n"
        f"from sklearn.{'linear_model' if task=='regression' else 'ensemble'} import "
        f"{'LinearRegression' if task=='regression' else 'RandomForestClassifier'}\n"
        "from sklearn.metrics import mean_squared_error, accuracy_score")]
    metric = "mean_squared_error(y_test, pred, squared=False)" if task=="regression" else "accuracy_score(y_test, pred)"
    cells += [nbf.v4.new_code_cell(
        f"X = df.select_dtypes(include=[np.number]).drop(columns=['{target}'], errors='ignore')\n"
        f"y = df['{target}'] if '{target}' in df.columns else df.iloc[:, -1]\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)  # no seed\n"
        f"model = {'LinearRegression()' if task=='regression' else 'RandomForestClassifier()'}\n"
        "model.fit(X_train, y_train)\n"
        "pred = model.predict(X_test)\n"
        f"print('Metric:', {metric})\n"
        "out = X_test.copy(); out['pred'] = pred\n"
        "out.to_csv('results.csv', index=False)  # overwrites prior file")]
    cells += [nbf.v4.new_markdown_cell("## Notes\n- TODO: imports first\n- TODO: remove absolute paths\n- TODO: set random seed\n- TODO: column naming consistency")]
    nb.cells = cells
    out_ipynb.write_text(nbf.writes(nb), encoding="utf-8")

def _make_nb_final(csv_name: str, out_ipynb: Path):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Analysis (FINAL??)"),
        nbf.v4.new_code_cell("import seaborn as sns  # pandas later"),
        nbf.v4.new_code_cell("import pandas as pd; df = pd.read_csv('" + csv_name + "')\n"
                             "df = df.sample(frac=1.0)  # shuffle without seed"),
        nbf.v4.new_code_cell("df.head()"),
        nbf.v4.new_code_cell("df.to_csv('results.csv', index=False)  # change schema again")
    ]
    out_ipynb.write_text(nbf.writes(nb), encoding="utf-8")

def generate(name: str, csv_path: str, target: str = None, task: str = "regression",
             difficulty: str = "medium", out_dir: str = None) -> Path:
    base_parent = Path(out_dir) if out_dir else Path.cwd()
    base = base_parent / f"{name}-messy"
    base.mkdir(parents=True, exist_ok=True)

    df, num_cols, cat_cols = _infer_columns(csv_path)
    if target is None:
        if task == 'regression' and num_cols:
            target = num_cols[-1]
        else:
            target = df.columns[-1]

    raw_csv = base / f"{name}.csv"
    _make_messy_csv(df, raw_csv, tweak_schema=(difficulty!='light'), inject_na=(difficulty!='light'))

    _make_nb_analysis(name, target, task, base / "analysis.ipynb")
    _make_nb_final(f"{name}.csv", base / "analysis (final).ipynb")

    _write_text(base / "helper.py", 
        """# Intentionally messy helpers using globals\n
def clean_data():\n
    df['tmp'] = df['tmp'] if 'tmp' in df else 0\n
    return df\n
""")
    _write_text(base / "plot.py",
        """import seaborn as sns, matplotlib.pyplot as plt\n
df = pd.read_csv('DATA.csv')  # pd not imported on purpose\n
sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1]); plt.savefig('figures/plot.png')\n
""")

    _write_text(base / "requirements.txt", "pandas==1.0.3\nseaborn\nmatplotlib==3.1.1\nscikit-learn\njupyter\n")
    _write_text(base / "environment.yml", 
        f"""name: {name}
channels: [conda-forge]
dependencies:
  - python=3.11
  - pandas=2.2
  - seaborn=0.13
  - matplotlib=3.8
  - scikit-learn=1.5
  - pip
  - pip:
      - jupyter==1.0.0
""")

    _write_text(base / "results.csv", "metric,value\nrmse,999.9\n")
    _write_text(base / "notes.txt", "TODO: set seed?? try random forest??\nData path: ~/Downloads/DATA.csv\n")
    _write_text(base / "README.md", f"# {name.title()}\nSome analysis. Work in progress.\n\nOpen the notebook and run some cells.\n")
    (base / "figures").mkdir(exist_ok=True)

    for p in base.rglob("*"):
        if p.suffix in {".py", ".md", ".txt", ".ipynb"}:
            try:
                s = p.read_text(encoding="utf-8")
                s = s.replace("DATA.csv", f"{name}.csv")
                p.write_text(s, encoding="utf-8")
            except Exception:
                pass

    return base
