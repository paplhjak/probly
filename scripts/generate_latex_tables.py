#!/usr/bin/env python3
"""Convert per-(dataset, loss) CSVs into four standalone LaTeX documents.

Reads ``experiments/epistemic_eval/results/csv/<dataset>__<loss>.csv``
(produced by ``generate_results_csvs.py``) and emits four
self-contained, separately-compileable documents under
``experiments/epistemic_eval/results/``:

* ``results_summary_dcic.tex`` -- cross-method summary aggregating
  over all DCIC ``(dataset, loss)`` cells (9 datasets x 2 losses).
* ``results_dcic.tex`` -- per-dataset detail tables for every DCIC
  dataset, both losses.
* ``results_appa_real.tex`` -- per-dataset detail table for APPA-REAL,
  both losses.
* ``results_cifar10h.tex`` -- per-dataset detail table for CIFAR-10H,
  both losses.

Each file is a complete ``\\documentclass{article}`` document with its
own preamble, macros, and legend so co-authors can compile any of
them in isolation. They share the same column layout, ``siunitx``
alignment, and colour conventions as the original combined document.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Iterable

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
CSV_DIR = _REPO_ROOT / "experiments" / "epistemic_eval" / "results" / "csv"
OUT_DIR = _REPO_ROOT / "experiments" / "epistemic_eval" / "results"

DCIC_DATASETS = [
    "benthic", "mice_bone", "pig", "plankton", "quality_mri",
    "dcic_synthetic", "treeversity_1", "treeversity_6", "turkey",
]

LOSS_LABELS = {
    "cross_entropy": "CE",
    "zero_one": r"$0/1$",
    "squared": "sq",
    "absolute": "abs",
}
METHOD_LABELS = {
    "mc_dropout": "MC-Dropout",
    "evidential": "Evidential",
    "ensemble":   "Ensemble",
    "ddu":        "DDU",
    "laplace":    "Laplace",
}
DATASET_LABELS = {
    "cifar10h":       "CIFAR-10H",
    "appa_real":      "APPA-REAL",
    "benthic":        "Benthic",
    "mice_bone":      "MiceBone",
    "pig":            "Pig",
    "plankton":       "Plankton",
    "quality_mri":    "QualityMRI",
    "dcic_synthetic": "DCIC-synth",
    "treeversity_1":  "Treeversity-1",
    "treeversity_6":  "Treeversity-6",
    "turkey":         "Turkey",
}

DATASET_LOSSES = {
    "appa_real": ("squared", "absolute"),
}
DEFAULT_LOSSES = ("cross_entropy", "zero_one")

DATASET_METHODS = {
    "appa_real": ("mc_dropout", "evidential", "ensemble", "ddu", "laplace"),
}
DEFAULT_METHODS = ("mc_dropout", "evidential", "ensemble", "ddu")

# (csv_key, header_label, siunitx table-format spec).
# Pareto-gap and rho_Ahat_Ehat are kept in the CSVs but excluded from
# the printed tables (per author preference); add them back here if
# desired.
#
# ``rho_Astar_regret`` in the CSV is the Spearman of (A*, realized
# regret). Under the paper's frequentist evaluation framework, E* per
# point IS the realized regret of the trained predictor (codebase's
# ``E_star = 0`` stored array is the oracle-of-itself convention, not
# the framework definition). So the column header reads
# ``r_s(A*, E*)``.
COLUMNS = [
    ("aurec",            r"$\AuReC$",                "2.4"),
    ("excess_aurec",     r"ex-$\AuReC$",             "2.4"),
    ("n_aurec",          r"$n$-$\AuReC$",            "1.3"),
    ("aurc",             r"$\AuRC$",                 "2.4"),
    # Pareto-gap: locally recomputed via the corrected frequentist
    # convention (E* = realised regret) by ``generate_results_csvs.py``,
    # not loaded from the cluster's metrics_<loss>.json.
    ("pareto_gap",       r"P-gap",                   "1.4"),
    # Three Spearman correlations.
    # rho_Ehat_regret  -> r_s(Ehat, E*)  since E* per-point equals
    #                     realized regret r_i in the frequentist
    #                     framework. This is the headline selector
    #                     quality.
    # rho_Ahat_Ehat    -> r_s(Ahat, Ehat) -- method-internal
    #                     decomposition entanglement.
    # rho_Astar_regret -> r_s(A*, E*) -- Bayes risk vs realized
    #                     regret across test points; high values mean
    #                     the trained model's regret tracks the
    #                     intrinsic noise structure.
    ("rho_Ehat_regret",  r"$r_s(\Ehat, \Estar)$",    "+1.3"),
    ("rho_Ahat_Ehat",    r"$r_s(\Ahat, \Ehat)$",     "+1.3"),
    ("rho_Astar_regret", r"$r_s(\Astar, \Estar)$",   "+1.3"),
]


# -------------------------- I/O helpers ------------------------------ #

def _load_csv(dataset: str, loss: str) -> list[dict]:
    path = CSV_DIR / f"{dataset}__{loss}.csv"
    return list(csv.DictReader(open(path))) if path.exists() else []


def _fnan(s: str) -> float:
    if s is None or s == "" or s == "nan":
        return float("nan")
    return float(s)


def _aggregate(rows: list[dict], key: str) -> tuple[float, float, int]:
    vals = [_fnan(r[key]) for r in rows
            if r.get("status") == "OK" and r.get(key) not in ("", "nan")]
    finite = [v for v in vals if not math.isnan(v)]
    n = len(finite)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(finite) / n
    std = statistics.stdev(finite) if n > 1 else 0.0
    return mean, std, n


# -------------------------- siunitx formatting ----------------------- #

def _value_cell(mean: float, std: float, fmt: str) -> str:
    """Render ``mean +/- std`` with the std in subscript-style smaller font.

    Format: ``$<mean>\\,{\\scriptscriptstyle\\pm\\,<std>}$``. Decimal
    places follow ``fmt`` (the digit count after the ``.`` in the
    table-format spec). The std uses the same precision as the mean so
    columns visually align even though we use plain ``r`` columns
    rather than siunitx's ``S``. The smaller std font keeps cells
    compact for paper-style tables.
    """
    if math.isnan(mean) or std is None or math.isnan(std):
        return r"\textcolor{gray!50}{--}"
    after = int(fmt.split(".")[1]) if "." in fmt else 0
    return (
        f"${mean:.{after}f}\\,"
        f"{{\\scriptscriptstyle\\pm\\,{std:.{after}f}}}$"
    )


# -------------------------- summary table ---------------------------- #

def _emit_summary_table(
    by_dataset_loss: dict[tuple, list[dict]],
    cells: Iterable[tuple[str, str]],
    label_suffix: str,
    title_qualifier: str,
) -> str:
    """Per-method summary aggregating over the given (dataset, loss) cells.

    ``cells`` is the iterable of (dataset, loss) pairs to include.
    ``label_suffix`` is appended to the table label
    (``tab:results-summary-{suffix}``). ``title_qualifier`` is
    inserted into the caption to describe the cell scope (e.g.,
    ``"DCIC datasets"``).
    """
    cells = list(cells)
    n_cells_total = len(cells)
    methods = ("mc_dropout", "evidential", "ensemble", "ddu", "laplace")

    per: dict[str, dict] = defaultdict(dict)
    for ds, loss in cells:
        rows = by_dataset_loss.get((ds, loss), [])
        cell_means_n_aurec = {}
        for method in methods:
            mrows = [r for r in rows if r["method"] == method]
            n_aurec_mean, _, n_seeds = _aggregate(mrows, "n_aurec")
            rho_eEstar_mean, _, _ = _aggregate(mrows, "rho_Ehat_regret")    # r_s(Ehat, E*)
            rho_AhEh_mean,   _, _ = _aggregate(mrows, "rho_Ahat_Ehat")      # r_s(Ahat, Ehat)
            rho_AsEs_mean,   _, _ = _aggregate(mrows, "rho_Astar_regret")   # r_s(A*, E*)
            if n_seeds > 0:
                per[method][(ds, loss)] = {
                    "n_aurec":  n_aurec_mean,
                    "rho_eEs":  rho_eEstar_mean,
                    "rho_AhEh": rho_AhEh_mean,
                    "rho_AsEs": rho_AsEs_mean,
                }
                cell_means_n_aurec[method] = n_aurec_mean
        if cell_means_n_aurec:
            winner = min(cell_means_n_aurec, key=cell_means_n_aurec.get)
            per[winner].setdefault("__wins__", set()).add((ds, loss))

    body = []
    for method in methods:
        cells_in = {k: v for k, v in per[method].items() if k != "__wins__"}
        if not cells_in:
            continue  # method doesn't appear in any of the requested cells
        n_covered = len(cells_in)
        wins = per[method].get("__wins__", set())
        n_wins = len(wins)
        n_anti = sum(1 for v in cells_in.values() if v["n_aurec"] > 1.0)

        def _vals(key: str) -> list[float]:
            return [v[key] for v in cells_in.values() if not math.isnan(v[key])]

        n_aurec_vals = [v["n_aurec"] for v in cells_in.values()]
        m_n_aurec  = sum(n_aurec_vals) / len(n_aurec_vals) if n_aurec_vals else float("nan")
        rho_eEs    = _vals("rho_eEs")
        rho_AhEh   = _vals("rho_AhEh")
        rho_AsEs   = _vals("rho_AsEs")
        m_rho_eEs  = sum(rho_eEs)  / len(rho_eEs)  if rho_eEs  else float("nan")
        m_rho_AhEh = sum(rho_AhEh) / len(rho_AhEh) if rho_AhEh else float("nan")
        m_rho_AsEs = sum(rho_AsEs) / len(rho_AsEs) if rho_AsEs else float("nan")

        method_lbl = METHOD_LABELS[method]
        coverage_cell = f"{n_covered}/{n_cells_total}"
        wins_cell = str(n_wins)
        anti_cell = (f"\\textcolor{{red!70}}{{{n_anti}}}"
                     if n_anti > 0 else f"{n_anti}")
        n_aurec_cell  = "{--}" if math.isnan(m_n_aurec)  else f"{m_n_aurec:.3f}"
        rho_eEs_cell  = "{--}" if math.isnan(m_rho_eEs)  else f"{m_rho_eEs:+.3f}"
        rho_AhEh_cell = "{--}" if math.isnan(m_rho_AhEh) else f"{m_rho_AhEh:+.3f}"
        rho_AsEs_cell = "{--}" if math.isnan(m_rho_AsEs) else f"{m_rho_AsEs:+.3f}"

        body.append(
            f"{method_lbl} & {coverage_cell} & {wins_cell} & {anti_cell} "
            f"& {n_aurec_cell} & {rho_eEs_cell} & {rho_AhEh_cell} "
            f"& {rho_AsEs_cell} \\\\"
        )

    caption = (
        f"\\textbf{{Cross-method summary across {title_qualifier}.}} "
        f"Each cell is one $(\\text{{dataset}}, \\text{{loss}})$ "
        f"combination ({n_cells_total} cells in total). "
        "\\emph{Cov.}: cells with $\\geq 1$ OK seed. "
        "\\emph{Won}: cells where the method achieves the lowest mean "
        "$n$-$\\AuReC$ (best selector). "
        "\\emph{$n{>}1$}: cells where the method's selector is "
        "worse than random. Means are taken over the cells the "
        "method actually covered. Lower $n$-$\\AuReC$ better; higher "
        "$r_s$ better."
    )
    return (
        "\\begin{table}[!ht]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:results-summary-{label_suffix}}}\n"
        "\\begin{tabular}{l c c c S[table-format=1.3]"
        " S[table-format=+1.3] S[table-format=+1.3] S[table-format=+1.3]}\n"
        "\\toprule\n"
        "Method & Cov. & Won & $n{>}1$ & "
        "{$\\overline{n\\text{-}\\AuReC}$} & "
        "{$\\overline{r_s(\\Ehat, \\Estar)}$} & "
        "{$\\overline{r_s(\\Ahat, \\Ehat)}$} & "
        "{$\\overline{r_s(\\Astar, \\Estar)}$} \\\\\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


# -------------------------- per-dataset table ------------------------ #

def _status_label(rows: list[dict]) -> str:
    statuses = {r["status"] for r in rows}
    if "MISSING_RUN" in statuses:
        return "no run"
    if "NO_PREDICTIONS" in statuses:
        return "fit only"
    if "NO_DECOMPOSITION" in statuses or "NO_METRICS" in statuses:
        return "incomplete"
    return "missing"


def _emit_dataset_table(rows_by_loss: dict[str, list[dict]], dataset: str) -> str:
    methods = DATASET_METHODS.get(dataset, DEFAULT_METHODS)
    losses = DATASET_LOSSES.get(dataset, DEFAULT_LOSSES)

    # Plain r columns; cells render mean +/- small-font std as text in
    # math mode. Decimal alignment is approximate (visually consistent
    # within a column when magnitudes are similar) but robust across
    # the wide magnitude ranges between datasets.
    col_spec = f"l l {'r ' * len(COLUMNS)}".rstrip()
    header = (
        "Method & Loss & "
        + " & ".join("{" + lbl + "}" for _, lbl, _ in COLUMNS)
        + " \\\\"
    )

    body_lines: list[str] = []
    last_method: str | None = None
    for method in methods:
        for loss in losses:
            rows = [r for r in rows_by_loss.get(loss, []) if r["method"] == method]
            ok = [r for r in rows if r.get("status") == "OK"]
            n_ok = len(ok)

            method_lbl = METHOD_LABELS[method]
            method_cell = method_lbl if method != last_method else ""
            last_method = method
            loss_lbl = LOSS_LABELS[loss]

            if n_ok == 0:
                tag = _status_label(rows) if rows else "no run"
                # When there's nothing to display, replace the whole
                # numeric strip with a single centred status marker
                # spanning all metric columns.
                marker = (
                    f"\\multicolumn{{{len(COLUMNS)}}}{{c}}"
                    f"{{\\textcolor{{red!70}}{{\\textsf{{{tag}}}}}}}"
                )
                body_lines.append(
                    " & ".join([method_cell, loss_lbl, marker]) + " \\\\"
                )
                continue

            cells = []
            for key, _, fmt in COLUMNS:
                vals = [_fnan(r[key]) for r in ok]
                finite = [v for v in vals if not math.isnan(v)]
                if not finite:
                    cells.append(r"\textcolor{gray!50}{--}")
                    continue
                mean = sum(finite) / len(finite)
                std = statistics.stdev(finite) if len(finite) > 1 else 0.0
                cells.append(_value_cell(mean, std, fmt))

            body_lines.append(
                " & ".join([method_cell, loss_lbl, *cells]) + " \\\\"
            )
        body_lines.append("\\addlinespace[2pt]")
    if body_lines and body_lines[-1].startswith("\\addlinespace"):
        body_lines.pop()

    caption = (
        f"\\textbf{{{DATASET_LABELS[dataset]}}} "
        f"(\\texttt{{{dataset.replace('_', '\\_')}}}). "
        "Each cell is mean$\\,\\pm\\,$std across seeds (std rendered "
        "smaller for compactness); lower is better for area metrics, "
        "higher better for $r_s$."
    )
    return (
        "\\begin{table}[!ht]\n"
        "\\centering\n"
        "\\footnotesize\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:results-{dataset.replace('_', '-')}}}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


# -------------------------- document assembly ------------------------ #

# Header banner placed at the top of each generated snippet file.
def _file_header(scope: str) -> str:
    return (
        f"% Auto-generated by scripts/generate_latex_tables.py\n"
        f"% Scope: {scope}\n"
        f"% Intended for \\input{{}} into a paper. To preview standalone,\n"
        f"% compile experiments/epistemic_eval/results/results_preview.tex\n"
        f"% which wraps this snippet (and the others) in a complete\n"
        f"% document.\n"
    )

# Macros + sisetup. Each output snippet starts with this so the
# macros are defined when the snippet is included into any parent
# document; ``\providecommand`` is a no-op when the parent already
# defines the symbol, so the paper's own definitions take precedence.
# Definitions below mirror the project's macros file exactly (spaces,
# predictors, distributions, uncertainty quantities, reject-option
# operators, math operators, eqdef, indicator).
#
# The parent paper still needs to load the LaTeX packages:
#   siunitx, booktabs, xcolor, colortbl, graphicx, amsmath.
PROVIDE_MACROS = r"""% --- Required packages in the surrounding paper -----------------------------
%   \usepackage{siunitx}    \usepackage{booktabs}   \usepackage{xcolor}
%   \usepackage{colortbl}   \usepackage{graphicx}   \usepackage{amsmath}
%
% --- Spaces and basic objects -----------------------------------------------
\providecommand{\Xspace}{\mathcal{X}}
\providecommand{\Yspace}{\mathcal{Y}}
\providecommand{\Hspace}{\mathcal{H}}
\providecommand{\Data}{\mathcal{D}}
\providecommand{\loss}{\ell}
\providecommand{\Reals}{\mathbb{R}}
\providecommand{\Rpos}{\mathbb{R}_{\geq 0}}
% --- Predictors -------------------------------------------------------------
\providecommand{\learnedH}{H}
\providecommand{\bayesH}{H^{*}}
\providecommand{\selector}{C}
% --- Distributions ----------------------------------------------------------
\providecommand{\ptrue}{p^{*}}
\providecommand{\predDist}[2]{p(#1 \mid #2)}
% --- Uncertainty quantities -------------------------------------------------
\providecommand{\Tstar}{T^{*}_{\ell}}
\providecommand{\Astar}[1][\ell]{A^{*}_{#1}}
\providecommand{\Estar}[1][\ell]{E^{*}_{#1}}
\providecommand{\Ahat}{\hat{A}_{\ell}}
\providecommand{\Ehat}{\hat{E}_{\ell}}
\providecommand{\That}{\hat{T}_{\ell}}
% --- Reject-option quantities -----------------------------------------------
\providecommand{\coverage}{\rho}
\providecommand{\Risk}{Ri}
\providecommand{\Regret}{Re}
\providecommand{\sRisk}{SRi}
\providecommand{\sRegret}{SRe}
\providecommand{\AuReC}{\mathrm{AuReC}}
\providecommand{\AuRC}{\mathrm{AURC}}
% --- Operators --------------------------------------------------------------
\providecommand{\E}{\mathbb{E}}
\providecommand{\Var}{\mathrm{Var}}
\providecommand{\Cov}{\mathrm{Cov}}
\providecommand{\Cor}{\mathrm{Cor}}
\providecommand{\argmin}{\mathop{\mathrm{arg\,min}}}
\providecommand{\argmax}{\mathop{\mathrm{arg\,max}}}
% --- Convenience ------------------------------------------------------------
\providecommand{\eqdef}{\triangleq}
\providecommand{\indicator}[1]{\mathbf{1}\!\left\{#1\right\}}

\sisetup{
  separate-uncertainty    = true,
  table-align-uncertainty = true,
  uncertainty-mode        = separate,
  detect-weight           = true,
}
"""

LEGEND = r"""\section*{Conventions and legend}

\begin{tabular}{ll}
\textcolor{gray!70}{\itshape --} & no run dir for this (method, seed) \\
\textcolor{red!70}{\textsf{no run}} & cluster grid hasn't reached this cell yet \\
\textcolor{red!70}{\textsf{fit only}} & method.pth written but extract did not run \\
\textcolor{red!70}{\textsf{incomplete}} & extract done, decomposition or metrics missing \\
\end{tabular}

\bigskip
\noindent
\textbf{Notation.} Per-point oracle quantities: $\Astar(x_i) \eqdef
\min_{y^{*} \in \Yspace} \E_{y \sim \ptrue(\cdot \mid x_i)}[\loss(y^{*}, y)]$
(irreducible loss given $\ptrue$);
$\Estar(x_i) \eqdef \E_{y \sim \ptrue(\cdot \mid x_i)}[\loss(\learnedH(x_i), y)] - \Astar(x_i)$
the per-point realised regret of the trained predictor $\learnedH$
relative to the Bayes-optimal predictor $\bayesH$;
$\Tstar(x_i) \eqdef \Astar(x_i) + \Estar(x_i)$ the total realised risk.
Method estimates are $\Ahat$, $\Ehat$ ($\Ehat$ is the selector
score). Coverage curves use the joint-expectation form of
Jaeger, Traub et al.\ (2024):
$\Risk(\coverage) \eqdef \frac{1}{N}\sum_{i \in \mathcal{A}_{\coverage}} \Tstar(x_i)$
and
$\Regret(\coverage) \eqdef \frac{1}{N}\sum_{i \in \mathcal{A}_{\coverage}} \Estar(x_i)$;
$\AuRC$ and $\AuReC$ are the integrals over $\coverage \in [0, 1]$.
$\text{ex-}\AuReC \eqdef \AuReC - \AuReC_{\mathrm{oracle}}$ subtracts
the irreducible component achievable with perfect ranking by
$\Estar$; $n\text{-}\AuReC \eqdef
(\AuReC - \AuReC_{\mathrm{oracle}}) / (\AuReC_{\mathrm{rand}} - \AuReC_{\mathrm{oracle}})$
with $\AuReC_{\mathrm{rand}} = \tfrac{1}{2}\,\E[\Estar]$, so $0=$ oracle,
$1=$ random, $>1=$ anti-correlated. $\AuReC_{\mathrm{oracle}}$
uses the learned predictor's regrets; only the ordering is optimal.
\smallskip
The column $r_s(\Astar, \Estar)$ is the Spearman correlation between
irreducible aleatoric loss and realised regret across test points;
high values indicate the predictor's residual error tracks the
dataset's intrinsic noise structure rather than a separate
model-side deficiency. Spearman correlations are pointwise over the
test set.

\bigskip
"""


def _snippet(scope: str, body: str) -> str:
    """An ``\\input``-friendly snippet: header + macros + table content.

    No ``\\documentclass`` / ``\\begin{document}`` / ``\\maketitle``;
    the surrounding paper provides those.
    """
    return _file_header(scope) + PROVIDE_MACROS + "\n" + body


def _preview_wrapper(snippet_files: list[str]) -> str:
    """Standalone document that ``\\input``s the snippets for a sanity
    compile. Lets you produce a single PDF for review without copying
    anything into a paper.
    """
    inputs = "\n".join(f"\\input{{{f}}}\n\\clearpage" for f in snippet_files)
    return (
        "% Auto-generated wrapper for previewing the snippet outputs.\n"
        "% Compile with: pdflatex results_preview.tex\n"
        "\\documentclass[10pt,a4paper]{article}\n"
        "\\usepackage[margin=1.8cm]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{array}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{colortbl}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath, amssymb}\n"
        "\\usepackage{siunitx}\n"
        "\\title{Empirical results -- preview wrapper}\n"
        "\\author{}\n\\date{}\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        + LEGEND
        + "\n"
        + inputs
        + "\n\\end{document}\n"
    )


# -------------------------- main ------------------------------------ #

def main() -> int:
    by_dataset_loss: dict[tuple, list[dict]] = {}
    all_datasets = DCIC_DATASETS + ["appa_real", "cifar10h"]
    for ds in all_datasets:
        for loss in DATASET_LOSSES.get(ds, DEFAULT_LOSSES):
            by_dataset_loss[(ds, loss)] = _load_csv(ds, loss)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. DCIC summary -- one block per loss so co-authors can compare
    #    methods within a fixed loss.
    summary_blocks: list[str] = []
    for loss in DEFAULT_LOSSES:
        cells_this_loss = [(ds, loss) for ds in DCIC_DATASETS
                           if (ds, loss) in by_dataset_loss]
        if not cells_this_loss:
            continue
        loss_label = {
            "cross_entropy": "cross-entropy",
            "zero_one": "$0/1$",
        }.get(loss, loss)
        summary_blocks.append(
            _emit_summary_table(
                by_dataset_loss,
                cells=cells_this_loss,
                label_suffix=f"dcic-{loss}",
                title_qualifier=f"DCIC datasets, {loss_label} loss",
            )
        )
    p = OUT_DIR / "results_summary_dcic.tex"
    p.write_text(_snippet(
        "Cross-method summary over DCIC datasets, per-loss",
        "\n".join(summary_blocks),
    ))
    written.append(p)

    # 2. DCIC per-dataset detail (one snippet, all 9 datasets).
    dcic_blocks: list[str] = []
    for ds in DCIC_DATASETS:
        rows_by_loss = {
            loss: by_dataset_loss[(ds, loss)]
            for loss in DATASET_LOSSES.get(ds, DEFAULT_LOSSES)
            if (ds, loss) in by_dataset_loss
        }
        if not rows_by_loss:
            continue
        dcic_blocks.append(_emit_dataset_table(rows_by_loss, ds))
    p = OUT_DIR / "results_dcic.tex"
    p.write_text(_snippet("Per-dataset results -- DCIC (9 datasets)",
                          "\n".join(dcic_blocks)))
    written.append(p)

    # 3. APPA-REAL.
    appa_rows = {
        loss: by_dataset_loss[("appa_real", loss)]
        for loss in DATASET_LOSSES.get("appa_real", DEFAULT_LOSSES)
        if ("appa_real", loss) in by_dataset_loss
    }
    p = OUT_DIR / "results_appa_real.tex"
    p.write_text(_snippet("Per-dataset results -- APPA-REAL",
                          _emit_dataset_table(appa_rows, "appa_real")))
    written.append(p)

    # 4. CIFAR-10H.
    cifar_rows = {
        loss: by_dataset_loss[("cifar10h", loss)]
        for loss in DATASET_LOSSES.get("cifar10h", DEFAULT_LOSSES)
        if ("cifar10h", loss) in by_dataset_loss
    }
    p = OUT_DIR / "results_cifar10h.tex"
    p.write_text(_snippet("Per-dataset results -- CIFAR-10H",
                          _emit_dataset_table(cifar_rows, "cifar10h")))
    written.append(p)

    # Standalone preview wrapper that pulls the four snippets together
    # for sanity-compiling. Not intended for inclusion in a paper.
    snippet_names = [
        "results_summary_dcic.tex",
        "results_cifar10h.tex",
        "results_appa_real.tex",
        "results_dcic.tex",
    ]
    p = OUT_DIR / "results_preview.tex"
    p.write_text(_preview_wrapper(snippet_names))
    written.append(p)

    print("wrote:")
    for p in written:
        print(f"  {p.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
