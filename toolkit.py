"""
One function per model. Each one loads a CSV, fits the model and writes an HTML report
to a reports/ folder next to the CSV.

    from toolkit import ols, pcr, pls

    ols('diabetes.csv', classes = ['Low', 'Moderate', 'High'], drop = ['Patient_ID'])
    pls('diabetes.csv', classes = ['Low', 'Moderate', 'High'], n_components = 5)

Settings every model accepts
    target       column to predict (default: the last column)
    classes      for a text target, its labels in order, e.g. ['Low', 'Moderate', 'High'].
                 Regressions turn them into 0, 1, 2 ...; classifiers use them as the label order.
    drop         columns to leave out (IDs, anything computed from the answer)
    predictors   use only these columns (default: every column except target and drop)
    test_size    share of rows held back for the test set (default 0.2)
    seed         random seed for the split and the CV folds (default 42)
    folds        number of cross-validation folds (default 5)
    report       file name for the report (default: <csv name>_<model>.html)
    show         open the report in the browser when it is written (default True)

Tuning parameters all follow one rule: leave it out for the default search, pass a list to
search over those values, or pass one number to fix it, e.g. lasso(..., alpha = [0.001, 0.01]).

Every function returns a Result: result.metrics, result.coefficients, result.model (fitted on
result.data.x_train_s, the standardised training predictors), result.report (the HTML path).

Adding a model: write a function that takes `data` (a Data object) plus its own parameters,
fits on data.x_train_s / data.y_train, and returns a Result with the test predictions.
Put @model('regression') or @model('classification') above it. ols() is the shortest example.
The report, metrics and standard charts are added for you.
"""

import base64
import functools
import html
import inspect
import io
import os
import sys
import warnings
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import (LinearRegression, LogisticRegression, LassoCV, RidgeCV, ElasticNetCV, Ridge,
                                  lasso_path, enet_path)
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, recall_score,
                             ConfusionMatrixDisplay)

# numpy 2.0 with Apple Accelerate prints false "encountered in matmul" warnings; the env var covers the parallel workers
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
warnings.filterwarnings('ignore', category = RuntimeWarning)

TEMPLATE = Path(__file__).parent / 'template.html'
MODELS = {}

# Text columns with more distinct values than this (names, IDs, free text) are left out instead of one-hot encoded
MAX_LEVELS = 50

# Categorical colours are used in this fixed order; text stays in the ink colours, never a series colour
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
BLUES = LinearSegmentedColormap.from_list('blues', [SURFACE, '#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])

STYLE = {
    'figure.facecolor' : SURFACE, 'axes.facecolor' : SURFACE, 'savefig.facecolor' : SURFACE,
    'axes.edgecolor' : GRID, 'axes.linewidth' : 1, 'axes.spines.top' : False, 'axes.spines.right' : False,
    'axes.grid' : True, 'grid.color' : GRID, 'grid.linewidth' : 0.8, 'axes.axisbelow' : True,
    'axes.titlesize' : 12, 'axes.titleweight' : 'semibold', 'axes.titlelocation' : 'left', 'axes.titlecolor' : INK,
    'axes.titlepad' : 10, 'axes.labelcolor' : INK_2, 'axes.labelsize' : 10,
    'xtick.color' : GRID, 'ytick.color' : GRID, 'xtick.labelcolor' : INK_2, 'ytick.labelcolor' : INK_2,
    'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
    'axes.prop_cycle' : cycler(color = SERIES), 'lines.linewidth' : 2, 'lines.solid_capstyle' : 'round',
    'legend.frameon' : False, 'legend.fontsize' : 9, 'legend.labelcolor' : INK_2,
    'font.size' : 10, 'text.color' : INK,
}


##############################################################################################################
#####                                         DATA AND RESULTS                                           #####
##############################################################################################################

@dataclass
class Data :
    source : Path
    raw : pd.DataFrame            # the CSV exactly as loaded
    target : str
    classes : list                # class labels in order, or None for a numeric target
    x_train : pd.DataFrame        # encoded and filled, original units
    x_test : pd.DataFrame
    x_train_s : pd.DataFrame      # standardised with the training mean and sd
    x_test_s : pd.DataFrame
    y_train : pd.Series           # regressions: numbers (classes become 0, 1, 2 ...); classifiers: labels
    y_test : pd.Series
    cv : object
    info : dict = field(default_factory = dict)
    notes : list = field(default_factory = list)

    @property
    def predictors(self) :
        return list(self.x_train.columns)


@dataclass
class Result :
    name : str                    # report title, e.g. 'PCR (M = 7)'
    y_pred : object = None        # predictions on the test set
    y_train_pred : object = None  # predictions on the training set (for train R² / accuracy)
    coefficients : pd.DataFrame = None  # one row per predictor; a 'Coefficient' first column gets a bar chart
    chosen : dict = field(default_factory = dict)    # what the model picked itself, e.g. the best alpha
    figures : list = field(default_factory = list)   # (caption, Figure)
    tables : list = field(default_factory = list)    # (title, DataFrame)
    notes : list = field(default_factory = list)
    model : object = None
    metrics : dict = field(default_factory = dict)
    data : Data = None
    report : Path = None


def _find(csv) :
    path = Path(csv).expanduser()
    if path.exists() :
        return path.resolve()
    # Fall back to the folder of the script being run, so 'diabetes.csv' works whatever the terminal's folder is
    main = getattr(sys.modules['__main__'], '__file__', None)
    if main and (Path(main).parent / path).exists() :
        return (Path(main).parent / path).resolve()
    raise FileNotFoundError(f"Can't find {csv} in {Path.cwd()}" + (f" or {Path(main).parent}" if main else ''))


def prepare(csv, kind, target = None, classes = None, drop = (), predictors = None, test_size = 0.2, seed = 42, folds = 5) :
    source = _find(csv)
    raw = pd.read_csv(source)
    target = target or raw.columns[-1]
    drop = list(drop)

    unknown = [c for c in [target, *drop, *(predictors or [])] if c not in raw.columns]
    if unknown :
        raise ValueError(f"Not columns of {source.name}: {unknown}. Columns are: {list(raw.columns)}")

    notes = []
    df = raw.dropna(subset = [target])
    if len(df) < len(raw) :
        notes.append(f"{len(raw) - len(df):,} rows with no {target} were left out.")

    y = df[target]
    x = df[list(predictors)] if predictors else df.drop(columns = [target, *drop])

    if classes is None and (kind == 'classification' or not pd.api.types.is_numeric_dtype(y)) :
        if kind == 'regression' :
            labels = ', '.join(map(str, sorted(y.unique(), key = str)[:10]))
            raise ValueError(f"{target} holds labels ({labels}). Pass classes = [...] from lowest to highest "
                             f"so they can be turned into 0, 1, 2 ...")
        classes = sorted(y.unique(), key = str)
    if classes is not None :
        classes = list(classes)
        extra = set(y.unique()) - set(classes)
        if extra :
            raise ValueError(f"{target} has labels missing from classes: {sorted(extra, key = str)}")
        if kind == 'regression' :
            y = y.map({c : i for i, c in enumerate(classes)})

    empty = list(x.columns[x.isna().all()])
    text = x.select_dtypes(exclude = ['number', 'bool']).columns
    too_many = [c for c in text if x[c].nunique() > MAX_LEVELS]
    if empty :
        notes.append(f"Left out because they are completely empty: {', '.join(empty)}.")
    if too_many :
        notes.append(f"Left out because they have more than {MAX_LEVELS} different text values: {', '.join(too_many)}. "
                     f"Pass them in drop to silence this.")
    x = x.drop(columns = empty + too_many)
    x = x.astype({c : int for c in x.select_dtypes('bool').columns})
    if x.shape[1] == 0 :
        raise ValueError('No predictor columns left to use.')

    stratify = y if classes is not None else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed, stratify = stratify)
    x_train, x_test = x_train.copy(), x_test.copy()

    num_cols = x.select_dtypes('number').columns
    cat_cols = x.select_dtypes(exclude = 'number').columns

    flagged = list(x_train.columns[x_train.isna().any()])
    for col in flagged :
        x_train[col + '_missing'] = x_train[col].isna().astype(int)
        x_test[col + '_missing'] = x_test[col].isna().astype(int)

    # Fill values come from the training set only
    medians = x_train[num_cols].median()
    x_train[num_cols] = x_train[num_cols].fillna(medians)
    x_test[num_cols] = x_test[num_cols].fillna(medians)
    if len(cat_cols) :
        modes = x_train[cat_cols].mode().iloc[0]
        x_train[cat_cols] = x_train[cat_cols].fillna(modes)
        x_test[cat_cols] = x_test[cat_cols].fillna(modes)

    x_train = pd.get_dummies(x_train, drop_first = True, dtype = int)
    x_test = pd.get_dummies(x_test, drop_first = True, dtype = int).reindex(columns = x_train.columns, fill_value = 0)

    scaler = StandardScaler().fit(x_train)
    x_train_s = pd.DataFrame(scaler.transform(x_train), index = x_train.index, columns = x_train.columns)
    x_test_s = pd.DataFrame(scaler.transform(x_test), index = x_test.index, columns = x_test.columns)

    if kind == 'classification' :
        cv = StratifiedKFold(n_splits = folds, shuffle = True, random_state = seed)
    else :
        cv = KFold(n_splits = folds, shuffle = True, random_state = seed)

    info = {
        'Rows in file' : len(raw),
        'Rows used' : len(df),
        'Training rows' : len(x_train),
        'Test rows' : len(x_test),
        'Columns used' : x.shape[1],
        'Predictors after encoding' : x_train.shape[1],
        'Text columns one-hot encoded' : len(cat_cols),
        'Missing-value flags added' : len(flagged),
        'Columns dropped' : drop or 'none',
    }
    return Data(source, raw, target, classes, x_train, x_test, x_train_s, x_test_s, y_train, y_test, cv, info, notes)


##############################################################################################################
#####                                   THE MODEL DECORATOR AND REPORTS                                  #####
##############################################################################################################

COMMON = ('target', 'classes', 'drop', 'predictors', 'test_size', 'seed', 'folds')
EYEBROW = {'regression' : 'Regression report', 'classification' : 'Classification report', 'explore' : 'Data exploration'}


def model(kind) :
    """Registers a fit function and wraps it so it can be called with a CSV path."""
    def register(fit) :
        @functools.wraps(fit)
        def run(csv, report = None, show = True, **settings) :
            common = {k : settings.pop(k) for k in COMMON if k in settings}
            params = _check_params(fit, settings)
            data = prepare(csv, kind, **common)
            with matplotlib.rc_context(STYLE) :
                result = _fit(fit, kind, data, settings)
                body = _model_body(result, _call(fit.__name__, csv, {**common, **settings}), params)
            result.report = _write(data, fit.__name__, kind, result.name, body, report)
            print(f"{result.name}: {_headline(result)}report saved to {result.report}")
            if show :
                webbrowser.open(result.report.as_uri())
            return result

        run.kind = kind
        MODELS[fit.__name__] = run
        return run
    return register


def _check_params(fit, settings) :
    """Fails fast on a misspelt parameter, before the CSV is loaded; returns every parameter with its default filled in."""
    signature = inspect.signature(fit)
    try :
        bound = signature.bind(None, **settings)
    except TypeError :
        allowed = [p for p in signature.parameters if p != 'data']
        bad = [k for k in settings if k not in allowed]
        raise TypeError(f"{fit.__name__}() doesn't take {bad}. Its own parameters are {allowed}; "
                        f"every model also takes {list(COMMON) + ['report', 'show']}.") from None
    bound.apply_defaults()
    return {k : v for k, v in bound.arguments.items() if k != 'data'}


def _fit(fit, kind, data, params) :
    result = fit(data, **params)
    result.data = data
    result.metrics = _score(data, result, kind)
    result.figures += _diagnostics(data, result, kind)
    return result


def _call(name, csv, settings) :
    args = ''.join(f", {k} = {v!r}" for k, v in settings.items())
    return f"{name}({str(csv)!r}{args})"


def _headline(result) :
    for key in ('Test R²', 'Balanced accuracy') :
        if key in result.metrics :
            return f"{key} {result.metrics[key]:.4f}, "
    return ''


def _subtitle(data, kind) :
    parts = [data.source.name]
    if kind != 'explore' :
        target = f"predicting {data.target}"
        if kind == 'regression' and data.classes :
            target += f" ({', '.join(f'{c} = {i}' for i, c in enumerate(data.classes))})"
        parts.append(target)
    parts.append(f"{len(data.x_train):,} training rows, {len(data.x_test):,} test rows")
    return ' · '.join(parts)


def _write(data, name, kind, title, body, report) :
    path = Path(report).expanduser() if report else Path(f"{data.source.stem}_{name}.html")
    if path.suffix != '.html' :
        path = path.with_suffix('.html')
    if not path.is_absolute() :
        path = data.source.parent / 'reports' / path
    path.parent.mkdir(parents = True, exist_ok = True)
    page = Template(TEMPLATE.read_text(encoding = 'utf-8')).substitute(
        title = html.escape(title),
        eyebrow = EYEBROW.get(kind, kind),
        subtitle = html.escape(_subtitle(data, kind)),
        body = body,
        generated = datetime.now().strftime('%d %b %Y, %H:%M'),
    )
    path.write_text(page, encoding = 'utf-8')
    return path


def _fmt(v) :
    if v is None :
        return '—'
    if isinstance(v, (bool, np.bool_)) :
        return str(bool(v))
    if isinstance(v, (int, np.integer)) :
        return f"{v:,}"
    if isinstance(v, (float, np.floating)) :
        if np.isnan(v) :
            return '—'
        if v != 0 and abs(v) < 1e-3 :
            return f"{v:.2e}"
        return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.4f}"
    if isinstance(v, (list, tuple, np.ndarray, range)) :
        return ', '.join(_fmt(i) for i in v)
    return str(v)


def _card(title, inner) :
    return f'<section class="card"><h2>{html.escape(title)}</h2>{inner}</section>\n'


def _tiles(metrics) :
    tiles = ''.join(f'<div class="tile"><div class="label">{html.escape(k)}</div><div class="value">{_fmt(v)}</div></div>'
                    for k, v in metrics.items())
    return f'<section class="tiles">{tiles}</section>\n'


def _kv_table(title, values) :
    rows = ''.join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(_fmt(v))}</td></tr>" for k, v in values.items())
    return f'<div><h3>{html.escape(title)}</h3><table class="kv"><tbody>{rows}</tbody></table></div>'


def _table(df) :
    return f'<div class="table-wrap">{df.to_html(border = 0, float_format = _fmt, na_rep = "—")}</div>'


def _figure_html(caption, fig) :
    buffer = io.BytesIO()
    fig.savefig(buffer, format = 'png', dpi = 144)
    image = base64.b64encode(buffer.getvalue()).decode()
    caption = html.escape(caption)
    width = min(fig.get_figwidth() / 11, 1) * 100
    return (f'<figure><img src="data:image/png;base64,{image}" alt="{caption}" style="width: {width:.0f}%">'
            f'<figcaption>{caption}</figcaption></figure>')


def _model_body(result, call, params) :
    data = result.data
    parts = [_tiles(result.metrics)] if result.metrics else []

    settings = {**{k : 'default' if v is None else v for k, v in params.items()}, **result.chosen}
    columns = _kv_table('Data', data.info) + (_kv_table('Settings', settings) if settings else '')
    parts.append(_card('Data and settings', f'<pre class="call">{html.escape(call)}</pre><div class="cols">{columns}</div>'))

    notes = data.notes + result.notes
    if notes :
        parts.append(_card('Notes', '<ul class="notes">' + ''.join(f"<li>{html.escape(n)}</li>" for n in notes) + '</ul>'))
    if result.figures :
        parts.append(_card('Charts', ''.join(_figure_html(caption, fig) for caption, fig in result.figures)))

    tables = ([('Coefficients', result.coefficients)] if result.coefficients is not None else []) + result.tables
    for title, df in tables :
        parts.append(_card(title, _table(df)))
    return ''.join(parts)


##############################################################################################################
#####                                     METRICS AND STANDARD CHARTS                                    #####
##############################################################################################################

def _rounded(y_pred, n_classes) :
    return np.clip(np.round(y_pred), 0, n_classes - 1).astype(int)


def _score(data, result, kind) :
    if result.y_pred is None :
        return {}
    y_pred = np.asarray(result.y_pred).ravel()
    y_test, y_train = data.y_test, data.y_train
    metrics = {}

    if kind == 'regression' :
        metrics['Test R²'] = r2_score(y_test, y_pred)
        metrics['Test MSE'] = mean_squared_error(y_test, y_pred)
        if result.y_train_pred is not None :
            metrics['Train R²'] = r2_score(y_train, np.asarray(result.y_train_pred).ravel())
        if data.classes :
            top = y_train.mode()[0]
            metrics['Accuracy (rounded)'] = accuracy_score(y_test, _rounded(y_pred, len(data.classes)))
            metrics[f"Baseline (always {data.classes[top]})"] = (y_test == top).mean()
    else :
        top = y_train.mode()[0]
        metrics['Balanced accuracy'] = balanced_accuracy_score(y_test, y_pred)
        metrics['Accuracy'] = accuracy_score(y_test, y_pred)
        if result.y_train_pred is not None :
            metrics['Train accuracy'] = accuracy_score(y_train, result.y_train_pred)
        recalls = recall_score(y_test, y_pred, labels = data.classes, average = None, zero_division = 0)
        metrics.update({f"Recall {c}" : r for c, r in zip(data.classes, recalls)})
        metrics[f"Baseline (always {top})"] = (y_test == top).mean()
    return metrics


def _figure(ncols = 1, width = 8, height = 4) :
    fig = Figure(figsize = (width, height), layout = 'constrained')
    return fig, fig.subplots(1, ncols)


def _confusion(ax, y_true, y_pred, labels, names, title, normalize = None) :
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels = labels, display_labels = names, normalize = normalize,
                                            cmap = BLUES, colorbar = False, ax = ax,
                                            values_format = '.2f' if normalize else ',d')
    ax.grid(False)
    ax.set_title(title)


def _class_boxplot(ax, y_true, y_pred, classes) :
    groups = [y_pred[y_true == k] for k in range(len(classes))]
    box = ax.boxplot(groups, tick_labels = [str(c) for c in classes], patch_artist = True, widths = 0.35,
                     medianprops = dict(color = INK, linewidth = 1.5),
                     flierprops = dict(marker = 'o', markersize = 2, markerfacecolor = MUTED, markeredgecolor = 'none', alpha = 0.4))
    for patch in box['boxes'] :
        patch.set(facecolor = SERIES[0] + '26', edgecolor = SERIES[0])
    for line in box['whiskers'] + box['caps'] :
        line.set(color = SERIES[0], linewidth = 1)
    for cut in np.arange(len(classes) - 1) + 0.5 :
        ax.axhline(cut, color = MUTED, linewidth = 1, linestyle = '--')
    ax.set_title('Predicted value by true class')
    ax.set_xlabel('True class')
    ax.set_ylabel('Predicted value')


def _coefficient_bars(coefficients, n = 15) :
    values = coefficients['Coefficient'].drop('(Intercept)', errors = 'ignore')
    top = values[values != 0].sort_values(key = abs).tail(n)
    if top.empty :
        return None
    fig, ax = _figure(height = 0.3 * len(top) + 1.2)
    ax.barh(top.index, top.values, height = 0.6, color = SERIES[0])
    ax.axvline(0, color = INK_2, linewidth = 0.8)
    ax.grid(axis = 'y', visible = False)
    ax.set_title(f"Largest {len(top)} coefficients")
    ax.set_xlabel('Coefficient')
    return fig


def _diagnostics(data, result, kind) :
    if result.y_pred is None :
        return []
    figures = []
    y_pred = np.asarray(result.y_pred).ravel()
    y_test = data.y_test.values

    if kind == 'regression' and data.classes :
        fig, (left, right) = _figure(2, width = 11, height = 4.2)
        _class_boxplot(left, y_test, y_pred, data.classes)
        codes = list(range(len(data.classes)))
        _confusion(right, y_test, _rounded(y_pred, len(codes)), codes, data.classes, 'Confusion matrix (rounded predictions)')
        figures.append(('Left: the dashed lines are where a prediction rounds to the next class. '
                        'Right: rows are the true class, columns the rounded prediction.', fig))
    elif kind == 'regression' :
        fig, (left, right) = _figure(2, width = 11, height = 4.2)
        left.scatter(y_test, y_pred, s = 6, alpha = 0.35, color = SERIES[0], linewidths = 0)
        low, high = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        left.plot([low, high], [low, high], color = INK_2, linewidth = 1, linestyle = '--')
        left.set_title('Predicted vs actual (test set)')
        left.set_xlabel('Actual')
        left.set_ylabel('Predicted')
        right.scatter(y_pred, y_test - y_pred, s = 6, alpha = 0.35, color = SERIES[0], linewidths = 0)
        right.axhline(0, color = INK_2, linewidth = 1)
        right.set_title('Residuals vs predicted')
        right.set_xlabel('Predicted')
        right.set_ylabel('Residual (actual − predicted)')
        figures.append(('Left: points on the dashed line are predicted exactly. Right: residuals should scatter evenly around zero.', fig))
    else :
        fig, (left, right) = _figure(2, width = 11, height = 4.2)
        _confusion(left, data.y_test, y_pred, data.classes, data.classes, 'Confusion matrix (counts)')
        _confusion(right, data.y_test, y_pred, data.classes, data.classes, 'Confusion matrix (share of each true class)', normalize = 'true')
        figures.append(('Rows are the true class, columns the prediction. The diagonal on the right is the recall of each class.', fig))

    if result.coefficients is not None and result.coefficients.columns[0] == 'Coefficient' :
        fig = _coefficient_bars(result.coefficients)
        if fig is not None :
            figures.append(('Coefficients furthest from zero, on the scale given in the notes.', fig))
    return figures


def _cv_plot(ax, xs, ys, best, title, xlabel, ylabel, log = False, label = None) :
    xs, ys = np.asarray(xs, dtype = float), np.asarray(ys, dtype = float)
    ax.plot(xs, ys, color = SERIES[0], marker = 'o' if len(xs) <= 60 else None, markersize = 3, label = label)
    best_y = ys[np.argmin(np.abs(xs - best))]
    ax.axvline(best, color = MUTED, linewidth = 1, linestyle = '--')
    ax.scatter([best], [best_y], s = 64, color = SERIES[1], edgecolors = SURFACE, linewidths = 2, zorder = 3)
    ax.annotate(f"Best = {_fmt(best)}", (best, best_y), xytext = (8, 8), textcoords = 'offset points', color = INK)
    if log :
        ax.set_xscale('log')
    ax.margins(y = 0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _path_plot(ax, xs, path, predictors, best = None, log = True, xlabel = 'Alpha (log scale)') :
    """Every coefficient along the penalty (or step) axis. The 8 largest at the chosen point get colours, the rest stay grey."""
    xs, path = np.asarray(xs, dtype = float), np.asarray(path)
    at = np.argmin(np.abs(xs - best)) if best is not None else -1
    top = sorted(range(len(predictors)), key = lambda i : (abs(path[i, at]), np.abs(path[i]).max()), reverse = True)[:8]
    for i in range(len(predictors)) :
        if i not in top :
            ax.plot(xs, path[i], color = GRID, linewidth = 1)
    for slot, i in enumerate(top) :
        ax.plot(xs, path[i], color = SERIES[slot], label = predictors[i])
    ax.axhline(0, color = INK_2, linewidth = 0.8)
    if best is not None :
        ax.axvline(best, color = MUTED, linewidth = 1, linestyle = '--')
    if log :
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Standardised coefficient')
    ax.legend(loc = 'upper left', bbox_to_anchor = (1.01, 1))


def _grid(value, default) :
    """None → the default search grid, a list → search over it, one number → fixed."""
    if value is None :
        return list(default)
    if isinstance(value, (list, tuple, np.ndarray, range)) :
        return list(value)
    return [value]


def _coefficients(values, predictors, intercept = None) :
    table = pd.DataFrame({'Coefficient' : np.asarray(values).ravel()}, index = list(predictors))
    table = table.sort_values('Coefficient', key = abs, ascending = False)
    if intercept is not None :
        table = pd.concat([pd.DataFrame({'Coefficient' : [float(np.ravel(intercept)[0])]}, index = ['(Intercept)']), table])
    return table


STANDARDISED = 'Coefficients are per standard deviation of each predictor, so their sizes can be compared directly.'


##############################################################################################################
#####                                             EXPLORING                                              #####
##############################################################################################################

@model('explore')
def explore(data) :
    """How the target is spread, which columns have gaps, and a summary of every column."""
    raw = data.raw
    result = Result(f"A first look at {data.source.name}")

    fig, ax = _figure(height = 3.6)
    if data.classes is not None :
        counts = raw[data.target].value_counts().reindex(data.classes)
        ax.bar([str(c) for c in counts.index], counts.values, width = 0.5, color = SERIES[0])
        for i, n in enumerate(counts.values) :
            ax.annotate(f"{n:,} ({n / counts.sum():.1%})", (i, n), xytext = (0, 4), textcoords = 'offset points', ha = 'center')
        ax.grid(axis = 'x', visible = False)
        ax.set_ylabel('Rows')
    else :
        ax.hist(raw[data.target].dropna(), bins = 40, color = SERIES[0], edgecolor = SURFACE)
        ax.set_ylabel('Rows')
        ax.set_xlabel(data.target)
    ax.set_title(f"{data.target}: how the rows are spread")
    result.figures.append(('A very uneven spread means plain accuracy is misleading; compare against the baseline.', fig))

    missing = raw.isna().mean().mul(100)
    missing = missing[missing > 0].sort_values()
    if len(missing) :
        fig, ax = _figure(height = 0.28 * len(missing) + 1.2)
        ax.barh(missing.index, missing.values, height = 0.6, color = SERIES[0])
        ax.grid(axis = 'y', visible = False)
        ax.set_title('Missing values by column')
        ax.set_xlabel('Percent of rows missing')
        result.figures.append(('Gaps are filled with the training median (numbers) or most common value (text), '
                               'and a _missing flag column records where they were.', fig))
    else :
        result.notes.append('No column has missing values.')

    result.tables.append(('Numeric columns', raw.select_dtypes('number').describe().T))
    text = raw.select_dtypes(exclude = 'number')
    if text.shape[1] :
        result.tables.append(('Text columns', pd.DataFrame({
            'Different values' : text.nunique(),
            'Most common' : [text[c].mode().iloc[0] if text[c].notna().any() else None for c in text],
            'Missing %' : text.isna().mean().mul(100),
        })))
    return result


@model('explore')
def pca(data) :
    """Principal components of the standardised predictors: scree plot, the first two components, and their loadings."""
    fit = PCA().fit(data.x_train_s)
    pve = fit.explained_variance_ratio_
    cum_pve = np.cumsum(pve)
    pcs = np.arange(1, len(pve) + 1)
    result = Result('Principal components', model = fit)
    result.chosen = {f"Components for {t:.0%} of variance" : int(np.argmax(cum_pve >= t)) + 1 for t in (0.80, 0.90, 0.95)}

    fig, (left, right) = _figure(2, width = 11, height = 4)
    left.plot(pcs, pve, color = SERIES[0], marker = 'o', markersize = 3)
    left.set_title('Scree plot')
    left.set_xlabel('Principal component')
    left.set_ylabel('Share of variance explained')
    right.plot(pcs, cum_pve, color = SERIES[0], marker = 'o', markersize = 3)
    for t in (0.80, 0.90, 0.95) :
        right.axhline(t, color = MUTED, linewidth = 0.8, linestyle = '--')
    right.set_title('Cumulative variance explained')
    right.set_xlabel('Number of components')
    right.set_ylabel('Cumulative share')
    result.figures.append((f"PC1 explains {pve[0]:.1%} and PC2 {pve[1]:.1%} of the variance in the predictors.", fig))

    scores = fit.transform(data.x_train_s)
    fig, ax = _figure(height = 5)
    if data.classes is not None :
        # Biggest class first, so the smaller ones are drawn on top
        for k in data.y_train.value_counts().index :
            mask = (data.y_train == k).values
            ax.scatter(scores[mask, 0], scores[mask, 1], s = 3, alpha = 0.3, linewidths = 0,
                       color = SERIES[data.classes.index(k) % len(SERIES)], label = str(k))
        ax.legend(markerscale = 5, title = data.target)
    else :
        points = ax.scatter(scores[:, 0], scores[:, 1], s = 3, alpha = 0.5, linewidths = 0, c = data.y_train, cmap = BLUES)
        fig.colorbar(points, ax = ax, label = data.target)
    ax.set_title('Training rows on the first two principal components')
    ax.set_xlabel(f"PC1 ({pve[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pve[1]:.1%})")
    result.figures.append(('If the classes separate here, a few components already carry most of the signal.', fig))

    loadings = pd.DataFrame(fit.components_[:2].T, index = data.predictors, columns = ['PC1', 'PC2'])
    fig, axes = _figure(2, width = 11, height = 4.5)
    for ax, pc in zip(axes, ['PC1', 'PC2']) :
        top = loadings[pc].sort_values(key = abs).tail(10)
        ax.barh(top.index, top.values, height = 0.6, color = SERIES[0])
        ax.axvline(0, color = INK_2, linewidth = 0.8)
        ax.grid(axis = 'y', visible = False)
        ax.set_title(f"Top 10 loadings on {pc}")
        ax.set_xlabel('Loading')
    result.figures.append(('The predictors that each of the first two components is mostly made of.', fig))

    result.tables.append(('Variance explained', pd.DataFrame({'Share' : pve, 'Cumulative' : cum_pve}, index = pd.Index(pcs, name = 'Component'))))
    result.tables.append(('Loadings on the first two components', loadings.sort_values('PC1', key = abs, ascending = False)))
    return result


##############################################################################################################
#####                                      LEAST SQUARES AND SUBSETS                                     #####
##############################################################################################################

@model('regression')
def ols(data, standardize = True) :
    """Ordinary least squares on every predictor, with standard errors and p-values."""
    x_train = data.x_train_s if standardize else data.x_train
    x_test = data.x_test_s if standardize else data.x_test
    fit = LinearRegression().fit(x_train, data.y_train)

    a = np.column_stack([np.ones(len(x_train)), x_train.values])
    beta = np.r_[fit.intercept_, fit.coef_]
    resid = data.y_train.values - a @ beta
    rank = np.linalg.matrix_rank(a)
    dof = len(a) - rank
    se = np.sqrt(np.diag(np.linalg.pinv(a.T @ a)) * (resid @ resid / dof))
    t = beta / se
    table = pd.DataFrame({'Coefficient' : beta, 'Std. error' : se, 't' : t, 'p-value' : 2 * stats.t.sf(np.abs(t), dof)},
                         index = ['(Intercept)', *data.predictors])
    table = pd.concat([table.iloc[:1], table.iloc[1:].sort_values('Coefficient', key = abs, ascending = False)])

    result = Result('OLS', fit.predict(x_test), fit.predict(x_train), coefficients = table, model = fit)
    result.notes.append(STANDARDISED if standardize else 'Coefficients are in the original units of each predictor.')
    if rank < a.shape[1] :
        result.notes.append('Some predictors are exact combinations of others, so their standard errors and p-values are not reliable.')

    # One predictor (ignoring its missing-value flag): draw the fitted line, as on Day 1
    original = [c for c in data.predictors if not c.endswith('_missing')]
    if len(original) == 1 and pd.api.types.is_numeric_dtype(data.raw[original[0]]) :
        col = original[0]
        order = np.argsort(data.x_test[col].values)
        fig, ax = _figure()
        ax.scatter(data.x_test[col], data.y_test, s = 6, alpha = 0.3, color = MUTED, linewidths = 0)
        ax.plot(data.x_test[col].values[order], result.y_pred[order], color = SERIES[0], linewidth = 0, marker = 'o', markersize = 2)
        ax.set_title(f"{data.target} against {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(data.target)
        result.figures.append(('Test rows in grey; the model\'s predictions in blue.', fig))
    return result


def _train_r2(x, y) :
    return LinearRegression().fit(x, y).score(x, y)


def _subset_row(data, cols, **extra) :
    fit = LinearRegression().fit(data.x_train_s[cols], data.y_train)
    y_pred = fit.predict(data.x_test_s[cols])
    return {
        'k' : len(cols),
        **extra,
        'CV R²' : cross_val_score(LinearRegression(), data.x_train_s[cols], data.y_train, cv = data.cv, scoring = 'r2').mean(),
        'Test R²' : r2_score(data.y_test, y_pred),
        'Test MSE' : mean_squared_error(data.y_test, y_pred),
        'Predictors' : list(cols),
    }


def _subset_result(name, data, table) :
    """k is chosen with cross-validation on the training set, so the test set is only used to report the final score."""
    table = table.sort_values('k').reset_index(drop = True)
    best = table.loc[table['CV R²'].idxmax()]
    k, cols = int(best['k']), best['Predictors']
    fit = LinearRegression().fit(data.x_train_s[cols], data.y_train)

    result = Result(f"{name} (k = {k})", fit.predict(data.x_test_s[cols]), fit.predict(data.x_train_s[cols]),
                    coefficients = _coefficients(fit.coef_, cols, fit.intercept_), model = fit)
    result.chosen = {'k (chosen by CV)' : k, 'Predictors used' : cols}
    result.notes.append(STANDARDISED)

    fig, ax = _figure()
    ax.plot(table['k'], table['Test R²'], color = SERIES[1], linewidth = 1.5, linestyle = ':', label = 'Test R² (reported only)')
    _cv_plot(ax, table['k'], table['CV R²'], k, f"{name}: R² vs number of predictors", 'Number of predictors (k)', 'R²',
             label = 'CV R² (used to choose k)')
    ax.legend(loc = 'lower right')
    result.figures.append(('k is chosen where the cross-validated R² peaks; the test R² is only reported, never used to choose.', fig))
    result.tables.append(('R² at every k', table.drop(columns = 'Predictors').set_index('k')))
    return result


@model('regression')
def top_k(data) :
    """Ranks predictors by their own training R², then adds them in that order; CV picks how many to keep."""
    singles = []
    for col in data.predictors :
        fit = LinearRegression().fit(data.x_train_s[[col]], data.y_train)
        y_pred = fit.predict(data.x_test_s[[col]])
        singles.append({'Predictor' : col, 'Coefficient' : fit.coef_[0], 'Train R²' : fit.score(data.x_train_s[[col]], data.y_train),
                        'Test R²' : r2_score(data.y_test, y_pred), 'Test MSE' : mean_squared_error(data.y_test, y_pred)})
    singles = pd.DataFrame(singles).sort_values('Train R²', ascending = False).set_index('Predictor')

    ranked = list(singles.index)
    rows = [_subset_row(data, ranked[:k], Added = ranked[k - 1]) for k in range(1, len(ranked) + 1)]
    result = _subset_result('Top-k', data, pd.DataFrame(rows))
    result.tables.append(('Each predictor on its own', singles))
    return result


@model('regression')
def forward_stepwise(data, max_k = None) :
    """Adds the predictor that raises training R² most, one at a time; CV picks how many to keep."""
    x, y = data.x_train_s, data.y_train
    max_k = max_k or len(data.predictors)
    selected, remaining, rows = [], list(data.predictors), []
    while len(selected) < max_k :
        best = max(remaining, key = lambda col : _train_r2(x[selected + [col]], y))
        selected.append(best)
        remaining.remove(best)
        rows.append(_subset_row(data, selected, Added = best))
    return _subset_result('Forward stepwise', data, pd.DataFrame(rows))


@model('regression')
def backward_stepwise(data, max_k = None) :
    """Starts with every predictor and drops the one whose loss hurts training R² least; CV picks how many to keep."""
    x, y = data.x_train_s, data.y_train
    max_k = max_k or len(data.predictors)
    selected, rows, dropped = list(data.predictors), [], None
    while True :
        if len(selected) <= max_k :
            rows.append(_subset_row(data, selected, Dropped = dropped or '—'))
        if len(selected) == 1 :
            break
        dropped = max(selected, key = lambda col : _train_r2(x[[c for c in selected if c != col]], y))
        selected.remove(dropped)
    return _subset_result('Backward stepwise', data, pd.DataFrame(rows))


##############################################################################################################
#####                                      SHRINKAGE: LASSO, RIDGE, ...                                  #####
##############################################################################################################

@model('regression')
def lasso(data, alpha = None) :
    """L1 penalty: shrinks coefficients and sets some exactly to zero. alpha is chosen by k-fold CV."""
    alphas = _grid(alpha, np.logspace(-4, 0, 100))
    fit = LassoCV(alphas = alphas, cv = data.cv).fit(data.x_train_s, data.y_train)
    result = Result(f"Lasso (alpha = {_fmt(fit.alpha_)})", fit.predict(data.x_test_s), fit.predict(data.x_train_s),
                    coefficients = _coefficients(fit.coef_, data.predictors, fit.intercept_), model = fit)
    result.chosen = {'alpha (chosen by CV)' : fit.alpha_, 'Predictors kept' : int((fit.coef_ != 0).sum())}
    result.notes.append(STANDARDISED)

    path_alphas = alphas if len(alphas) > 1 else np.logspace(-4, 0, 100)
    xs, path, _ = lasso_path(data.x_train_s.values, data.y_train.values, alphas = path_alphas)
    fig, ax = _figure(width = 9, height = 4.5)
    _path_plot(ax, xs, path, data.predictors, fit.alpha_)
    ax.set_title('Lasso coefficient path')
    result.figures.append(('Moving right, the penalty grows and coefficients shrink to zero. The dashed line is the chosen alpha.', fig))

    if len(alphas) > 1 :
        fig, ax = _figure()
        _cv_plot(ax, fit.alphas_, fit.mse_path_.mean(axis = 1), fit.alpha_, 'Lasso: CV MSE vs alpha', 'Alpha (log scale)', 'Mean CV MSE', log = True)
        result.figures.append(('alpha is chosen where the cross-validated error is lowest.', fig))
    return result


@model('regression')
def ridge(data, alpha = None) :
    """L2 penalty: shrinks every coefficient towards zero but keeps them all. alpha is chosen by leave-one-out CV."""
    alphas = _grid(alpha, np.logspace(-2, 5, 100))
    fit = RidgeCV(alphas = alphas, store_cv_results = True).fit(data.x_train_s, data.y_train)
    result = Result(f"Ridge (alpha = {_fmt(fit.alpha_)})", fit.predict(data.x_test_s), fit.predict(data.x_train_s),
                    coefficients = _coefficients(fit.coef_, data.predictors, fit.intercept_), model = fit)
    result.chosen = {'alpha (chosen by leave-one-out CV)' : fit.alpha_}
    result.notes.append(STANDARDISED)

    path_alphas = alphas if len(alphas) > 1 else np.logspace(-2, 5, 100)
    path = np.array([Ridge(alpha = a).fit(data.x_train_s.values, data.y_train.values).coef_ for a in path_alphas]).T
    fig, ax = _figure(width = 9, height = 4.5)
    _path_plot(ax, path_alphas, path, data.predictors, fit.alpha_)
    ax.set_title('Ridge coefficient path')
    result.figures.append(('Moving right, the penalty grows and every coefficient shrinks towards zero. The dashed line is the chosen alpha.', fig))

    if len(alphas) > 1 :
        fig, ax = _figure()
        _cv_plot(ax, alphas, fit.cv_results_.mean(axis = 0), fit.alpha_, 'Ridge: leave-one-out MSE vs alpha', 'Alpha (log scale)', 'Mean LOO MSE', log = True)
        result.figures.append(('alpha is chosen where the leave-one-out error is lowest.', fig))
    return result


@model('regression')
def elastic_net(data, alpha = None, l1_ratio = None) :
    """A mix of the lasso and ridge penalties. l1_ratio = 1 is pure lasso, 0 is pure ridge. Both are chosen by k-fold CV."""
    alphas = _grid(alpha, np.logspace(-4, 0, 100))
    ratios = _grid(l1_ratio, [0.1, 0.5, 0.9])
    fit = ElasticNetCV(alphas = alphas, l1_ratio = ratios, cv = data.cv).fit(data.x_train_s, data.y_train)
    result = Result(f"Elastic net (alpha = {_fmt(fit.alpha_)}, l1_ratio = {fit.l1_ratio_})", fit.predict(data.x_test_s),
                    fit.predict(data.x_train_s), coefficients = _coefficients(fit.coef_, data.predictors, fit.intercept_), model = fit)
    result.chosen = {'alpha (chosen by CV)' : fit.alpha_, 'l1_ratio (chosen by CV)' : fit.l1_ratio_,
                     'Predictors kept' : int((fit.coef_ != 0).sum())}
    result.notes.append(STANDARDISED)

    path_alphas = alphas if len(alphas) > 1 else np.logspace(-4, 0, 100)
    xs, path, _ = enet_path(data.x_train_s.values, data.y_train.values, alphas = path_alphas, l1_ratio = fit.l1_ratio_)
    fig, ax = _figure(width = 9, height = 4.5)
    _path_plot(ax, xs, path, data.predictors, fit.alpha_)
    ax.set_title(f"Elastic net coefficient path (l1_ratio = {fit.l1_ratio_})")
    result.figures.append(('Moving right, the penalty grows and coefficients shrink. The dashed line is the chosen alpha.', fig))

    if len(alphas) > 1 :
        # With several l1_ratios sklearn adds a leading axis for them; take the chosen one
        mse, cv_alphas = np.asarray(fit.mse_path_), np.asarray(fit.alphas_)
        if mse.ndim == 3 :
            mse = mse[ratios.index(fit.l1_ratio_)]
        if cv_alphas.ndim == 2 :
            cv_alphas = cv_alphas[ratios.index(fit.l1_ratio_)]
        fig, ax = _figure()
        _cv_plot(ax, cv_alphas, mse.mean(axis = 1), fit.alpha_, 'Elastic net: CV MSE vs alpha', 'Alpha (log scale)', 'Mean CV MSE', log = True)
        result.figures.append((f"Shown for the chosen l1_ratio of {fit.l1_ratio_}.", fig))
    return result


def _run_stagewise(x_s, y_values, eps, n_steps) :
    """Returns the mean of y and the weights after every step (one row per step)."""
    mean = y_values.mean()
    beta = np.zeros(x_s.shape[1])
    path = []
    for step in range(n_steps) :
        residual = y_values - mean - x_s @ beta
        corr = x_s.T @ residual
        j = np.argmax(np.abs(corr))
        beta[j] += eps * np.sign(corr[j])
        path.append(beta.copy())
    return mean, np.array(path)


@model('regression')
def stagewise(data, steps = None, eps = 0.001, max_steps = 2000) :
    """Forward stagewise: each step nudges the coefficient most correlated with the residual by eps. The number of steps is chosen by k-fold CV."""
    grid = _grid(steps, range(1, max_steps + 1))
    n_steps = max(grid)
    x_s, y = data.x_train_s.values, data.y_train.values

    cv_mse = np.zeros(n_steps)
    for fit_idx, val_idx in data.cv.split(x_s, y) :
        fold_mean, fold_path = _run_stagewise(x_s[fit_idx], y[fit_idx], eps, n_steps)
        val_pred = fold_mean + x_s[val_idx] @ fold_path.T
        cv_mse += ((val_pred - y[val_idx][:, None]) ** 2).mean(axis = 0) / data.cv.get_n_splits()
    best_step = min(grid, key = lambda s : cv_mse[s - 1])

    y_mean, path = _run_stagewise(x_s, y, eps, n_steps)
    beta = path[best_step - 1]
    result = Result(f"Forward stagewise ({best_step} steps)", y_mean + data.x_test_s.values @ beta, y_mean + x_s @ beta,
                    coefficients = _coefficients(beta, data.predictors, y_mean))
    result.chosen = {'Steps (chosen by CV)' : best_step, 'Predictors kept' : int((beta != 0).sum())}
    result.notes.append(STANDARDISED)

    fig, ax = _figure(width = 9, height = 4.5)
    _path_plot(ax, np.arange(1, n_steps + 1), path.T, data.predictors, best_step, log = False, xlabel = 'Step')
    ax.set_title('Stagewise coefficient path')
    result.figures.append(('Each step moves one coefficient by eps. The dashed line is the chosen number of steps.', fig))

    if len(grid) > 1 :
        fig, ax = _figure()
        _cv_plot(ax, grid, cv_mse[np.array(grid) - 1], best_step, 'Stagewise: CV MSE vs number of steps', 'Step', 'Mean CV MSE')
        result.figures.append(('The number of steps is chosen where the cross-validated error is lowest.', fig))
    return result


##############################################################################################################
#####                                     DIMENSION REDUCTION: PCR, PLS                                  #####
##############################################################################################################

def _components_cv(data, make, grid) :
    return np.array([-cross_val_score(make(m), data.x_train, data.y_train, scoring = 'neg_mean_squared_error', cv = data.cv).mean()
                     for m in grid])


@model('regression')
def pcr(data, n_components = None) :
    """Principal components regression: OLS on the first M principal components. M is chosen by k-fold CV."""
    p = len(data.predictors)
    grid = _grid(n_components, range(1, p + 1))
    make = lambda m : make_pipeline(StandardScaler(), PCA(n_components = m), LinearRegression())
    cv_mse = _components_cv(data, make, grid) if len(grid) > 1 else None
    best = grid[int(np.argmin(cv_mse))] if cv_mse is not None else grid[0]

    fit = make(best).fit(data.x_train, data.y_train)
    pca_step, lr = fit.named_steps['pca'], fit.named_steps['linearregression']
    # Map the component weights back onto the standardised predictors
    coefs = pca_step.components_.T @ lr.coef_
    result = Result(f"PCR (M = {best})", fit.predict(data.x_test), fit.predict(data.x_train),
                    coefficients = _coefficients(coefs, data.predictors, lr.intercept_), model = fit)
    result.chosen = {'Components M' + (' (chosen by CV)' if cv_mse is not None else '') : best,
                     'Variance of predictors explained' : pca_step.explained_variance_ratio_.sum()}
    result.notes.append(STANDARDISED + ' They are the component weights mapped back onto the original predictors.')

    if cv_mse is not None :
        fig, ax = _figure()
        _cv_plot(ax, grid, cv_mse, best, 'PCR: CV MSE vs number of components', 'Number of principal components (M)', 'Mean CV MSE')
        ax.axhline(np.var(data.y_train), color = INK_2, linewidth = 1, linestyle = ':')
        ax.annotate('Predict the mean', (grid[-1], np.var(data.y_train)), xytext = (-4, 4), textcoords = 'offset points', ha = 'right', color = INK_2)
        result.figures.append(('M is chosen where the cross-validated error is lowest.', fig))
        result.tables.append(('CV MSE by number of components', pd.DataFrame({'CV MSE' : cv_mse}, index = pd.Index(grid, name = 'M'))))
    return result


@model('regression')
def pls(data, n_components = None, max_components = 30) :
    """Partial least squares: components chosen to explain y, not just the predictors. M is chosen by k-fold CV."""
    p = len(data.predictors)
    grid = _grid(n_components, range(1, min(p, max_components) + 1))
    make = lambda m : make_pipeline(StandardScaler(), PLSRegression(n_components = m, scale = False))
    cv_mse = _components_cv(data, make, grid) if len(grid) > 1 else None
    best = grid[int(np.argmin(cv_mse))] if cv_mse is not None else grid[0]

    fit = make(best).fit(data.x_train, data.y_train)
    pls_step = fit.named_steps['plsregression']
    result = Result(f"PLS (M = {best})", fit.predict(data.x_test).ravel(), fit.predict(data.x_train).ravel(),
                    coefficients = _coefficients(pls_step.coef_, data.predictors, pls_step.intercept_), model = fit)
    result.chosen = {'Components M' + (' (chosen by CV)' if cv_mse is not None else '') : best}
    result.notes.append(STANDARDISED)

    if cv_mse is not None :
        fig, ax = _figure()
        _cv_plot(ax, grid, cv_mse, best, 'PLS: CV MSE vs number of components', 'Number of PLS components (M)', 'Mean CV MSE')
        ax.axhline(np.var(data.y_train), color = INK_2, linewidth = 1, linestyle = ':')
        ax.annotate('Predict the mean', (grid[-1], np.var(data.y_train)), xytext = (-4, 4), textcoords = 'offset points', ha = 'right', color = INK_2)
        result.figures.append(('M is chosen where the cross-validated error is lowest.', fig))
        result.tables.append(('CV MSE by number of components', pd.DataFrame({'CV MSE' : cv_mse}, index = pd.Index(grid, name = 'M'))))
    return result


##############################################################################################################
#####                                           CLASSIFIERS                                              #####
##############################################################################################################

@model('classification')
def logistic(data, C = None, balanced = True) :
    """Logistic regression. C is the inverse of the penalty strength, chosen by CV balanced accuracy."""
    grid = _grid(C, [0.001, 0.01, 0.1, 1, 10, 100])
    search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight = 'balanced' if balanced else None, max_iter = 2000)),
                          {'logisticregression__C' : grid}, scoring = 'balanced_accuracy', cv = data.cv, n_jobs = -1)
    search.fit(data.x_train, data.y_train)
    best = search.best_params_['logisticregression__C']
    lr = search.best_estimator_.named_steps['logisticregression']

    result = Result(f"Logistic regression (C = {_fmt(best)})", search.predict(data.x_test), search.predict(data.x_train), model = search)
    result.chosen = {'C (chosen by CV)' : best}
    result.notes.append(STANDARDISED)

    if len(lr.classes_) == 2 :
        result.coefficients = _coefficients(lr.coef_[0], data.predictors, lr.intercept_)
        result.notes.append(f"Positive coefficients push a row towards {lr.classes_[1]}.")
    else :
        classes = [str(c) for c in data.classes]
        weights = pd.DataFrame(lr.coef_.T, index = data.predictors, columns = [str(c) for c in lr.classes_])[classes]
        result.coefficients = weights.loc[weights.abs().max(axis = 1).sort_values(ascending = False).index]
        fig, axes = _figure(len(classes), width = min(11, 3.7 * len(classes)), height = 4.5)
        for ax, c in zip(np.atleast_1d(axes), classes) :
            top = weights[c].sort_values(key = abs).tail(10)
            ax.barh(top.index, top.values, height = 0.6, color = SERIES[0])
            ax.axvline(0, color = INK_2, linewidth = 0.8)
            ax.grid(axis = 'y', visible = False)
            ax.set_title(f"Top 10 for {c}")
        result.figures.append(('The predictors that push a row hardest towards (positive) or away from (negative) each class.', fig))

    if len(grid) > 1 :
        fig, ax = _figure()
        _cv_plot(ax, grid, search.cv_results_['mean_test_score'], best, 'Logistic regression: CV balanced accuracy vs C', 'C (log scale)', 'Mean CV balanced accuracy', log = True)
        result.figures.append(('C is chosen where cross-validated balanced accuracy is highest.', fig))
    return result


@model('classification')
def naive_bayes(data, balanced = True) :
    """Gaussian naive Bayes: treats every predictor as independent and normally distributed within each class."""
    weights = compute_sample_weight('balanced', data.y_train) if balanced else None
    fit = GaussianNB().fit(data.x_train, data.y_train, sample_weight = weights)
    result = Result('Naive Bayes', fit.predict(data.x_test), fit.predict(data.x_train), model = fit)
    if balanced :
        result.notes.append('Rows are weighted so every class counts equally in training.')
    return result


@model('classification')
def knn(data, k = None) :
    """k-nearest neighbours on standardised predictors. k is chosen by CV balanced accuracy."""
    grid = _grid(k, range(1, 51, 2))
    search = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), {'kneighborsclassifier__n_neighbors' : grid},
                          scoring = 'balanced_accuracy', cv = data.cv, n_jobs = -1)
    search.fit(data.x_train, data.y_train)
    best = search.best_params_['kneighborsclassifier__n_neighbors']
    result = Result(f"k-NN (k = {best})", search.predict(data.x_test), search.predict(data.x_train), model = search)
    result.chosen = {'k (chosen by CV)' : best}

    if len(grid) > 1 :
        fig, ax = _figure()
        _cv_plot(ax, grid, search.cv_results_['mean_test_score'], best, 'k-NN: CV balanced accuracy vs k', 'k (number of neighbours)', 'Mean CV balanced accuracy')
        result.figures.append(('k is chosen where cross-validated balanced accuracy is highest.', fig))
    return result


##############################################################################################################
#####                                       COMPARING SEVERAL MODELS                                     #####
##############################################################################################################

def compare(csv, models, report = None, show = True, **settings) :
    """Fits several models on the same train/test split and puts them side by side in one report.

        compare('diabetes.csv', [ols, lasso, ridge, (pcr, {'n_components' : 5})], classes = ['Low', 'Moderate', 'High'])

    Each entry is a model (or its name) or a (model, {its parameters}) pair. All must be regressions or all classifiers.
    """
    specs = []
    for entry in models :
        fn, params = entry if isinstance(entry, tuple) else (entry, {})
        name = fn if isinstance(fn, str) else fn.__name__
        if name not in MODELS or MODELS[name].kind == 'explore' :
            raise ValueError(f"{name} is not a model. Choose from: {[n for n, f in MODELS.items() if f.kind != 'explore']}")
        specs.append((name, dict(params)))
    kinds = {MODELS[name].kind for name, _ in specs}
    if len(kinds) != 1 :
        raise ValueError('compare() needs all regressions or all classifiers, not a mix.')
    kind = kinds.pop()

    common = {k : settings.pop(k) for k in COMMON if k in settings}
    if settings :
        raise TypeError(f"compare() doesn't take {list(settings)}. Put model parameters in a (model, {{...}}) pair.")
    fits = [MODELS[name].__wrapped__ for name, _ in specs]
    checked = [_check_params(fit, params) for fit, (_, params) in zip(fits, specs)]
    data = prepare(csv, kind, **common)

    results = []
    with matplotlib.rc_context(STYLE) :
        for fit, (name, params), full in zip(fits, specs, checked) :
            print(f"Fitting {name} ...")
            result = _fit(fit, kind, data, params)
            results.append((result, _model_body(result, _call(name, csv, {**common, **params}), full)))
        body = _compare_body(data, kind, [r for r, _ in results], _call('compare', csv, {'models' : [n for n, _ in specs], **common}))

    body += ''.join(f'<details class="model"><summary>{html.escape(r.name)}<span>{html.escape(_headline(r).rstrip(", "))}</span></summary>'
                    f'<div class="inner">{inner}</div></details>\n' for r, inner in results)
    title = 'Model comparison'
    path = _write(data, 'compare', kind, title, body, report)
    print(f"Comparison of {len(results)} models: report saved to {path}")
    if show :
        webbrowser.open(path.as_uri())
    for result, _ in results :
        result.report = path
    return [r for r, _ in results]


def _compare_body(data, kind, results, call) :
    headline = 'Test R²' if kind == 'regression' else 'Balanced accuracy'
    summary = pd.DataFrame([r.metrics for r in results], index = pd.Index([r.name for r in results], name = 'Model'))
    best = summary[headline].idxmax()
    parts = [_tiles({f"Best {headline}" : summary[headline].max(), 'Best model' : best, 'Models compared' : len(results)})]
    parts.append(_card('Data and settings', f'<pre class="call">{html.escape(call)}</pre><div class="cols">{_kv_table("Data", data.info)}</div>'))
    parts.append(_card('Scores on the same test set', _table(summary)))

    order = summary[headline].sort_values()
    fig, ax = _figure(height = 0.45 * len(order) + 1.2)
    ax.margins(x = 0.12)
    ax.barh(order.index, order.values, height = 0.6, color = SERIES[0])
    for i, v in enumerate(order.values) :
        ax.annotate(_fmt(v), (v, i), xytext = (4, 0), textcoords = 'offset points', va = 'center')
    ax.grid(axis = 'y', visible = False)
    ax.set_title(f"{headline} by model")
    ax.set_xlabel(headline)
    charts = [_figure_html('Every model is scored on the same held-out rows.', fig)]

    with_coefs = [r for r in results if r.coefficients is not None and r.coefficients.columns[0] == 'Coefficient']
    if with_coefs :
        table = pd.DataFrame({r.name : r.coefficients['Coefficient'] for r in with_coefs}).drop('(Intercept)', errors = 'ignore')
        table = table.loc[table.abs().max(axis = 1).sort_values(ascending = False).index]
        if len(with_coefs) <= 4 :
            top = table.head(15).iloc[::-1]
            fig, ax = _figure(height = 0.45 * len(top) + 1.5)
            band = 0.8 / len(with_coefs)
            for slot, name in enumerate(top.columns) :
                ax.barh(np.arange(len(top)) + (slot - (len(with_coefs) - 1) / 2) * band, top[name].fillna(0), height = band * 0.85,
                        color = SERIES[slot], label = name)
            ax.set_yticks(np.arange(len(top)), top.index)
            ax.axvline(0, color = INK_2, linewidth = 0.8)
            ax.grid(axis = 'y', visible = False)
            ax.legend(loc = 'lower right')
            ax.set_title('Largest coefficients by model')
            ax.set_xlabel('Coefficient')
            charts.append(_figure_html('Predictors with the largest coefficient in any model. Blank cells in the table '
                                       'are predictors a model did not use.', fig))
        parts.append(_card('Charts', ''.join(charts)))
        parts.append(_card('Coefficients by model', _table(table)))
    else :
        parts.append(_card('Charts', ''.join(charts)))
    parts.append('<section class="card"><h2>Each model in detail</h2><p class="subtitle">Click a model to open its full report.</p></section>\n')
    return ''.join(parts)
