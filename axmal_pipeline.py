"""
ml_pipeline.py
==============
Shortcut class untuk pipeline:
  1. Baseline (semua fitur, tanpa SMOTE)
  2. Feature Selection via SHAP (dari X_train)
  3. SMOTETomek resampling per persentil
  4. Komparasi lengkap Baseline vs Feature Selection

Penggunaan minimal:
    from ml_pipeline import MLPipeline
    pipe = MLPipeline(X_train, X_test, y_train, y_test)
    pipe.run()
    pipe.show_comparison()
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek


# ── Kelas Utama ───────────────────────────────────────────────────────────────
class MLPipeline:
    """
    Pipeline ML lengkap dengan SHAP feature selection dan komparasi Baseline.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Fitur training dan testing (sudah di-split).
    y_train, y_test : pd.Series
        Label training dan testing.
    percentiles : list[int], optional
        Persentase fitur yang dipilih dari ranking SHAP.
        Default: [10, 20, 30, 40, 50, 60, 70]
    shap_model : str, optional
        Model yang dipakai untuk menghitung SHAP ('xgboost' | 'lgbm' | 'rf').
        Default: 'xgboost'
    """

    # ── Default models ────────────────────────────────────────────────────────
    DEFAULT_MODELS = {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.2, random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.2, class_weight="balanced",
            random_state=42, verbose=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_split=5, class_weight="balanced",
            random_state=42
        ),
    }

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        percentiles: list = None,
        shap_model: str = "xgboost",
    ):
        self.X_train = X_train.reset_index(drop=True)
        self.X_test  = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test  = y_test.reset_index(drop=True)

        self.percentiles  = percentiles or [10, 20, 30, 40, 50, 60, 70]
        self.shap_model   = shap_model.lower()
        self.class_labels = sorted(y_train.unique())

        # Hasil yang diisi setelah run()
        self.shap_importance_df: pd.DataFrame = None
        self.baseline_results:   dict         = {}   # {model_name: metrics}
        self.fs_results:         dict         = {}   # {f"P{P}_{model}": metrics}
        self.results_df:         pd.DataFrame = None

    # ═════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════════════

    def run(self, plot_cm: bool = True, plot_shap: bool = True):
        """Jalankan seluruh pipeline secara berurutan."""
        print("\n" + "█"*60)
        print("  ML PIPELINE — BASELINE + FEATURE SELECTION")
        print("█"*60)

        self._step_shap(plot=plot_shap)
        self._step_baseline()
        self._step_feature_selection(plot_cm=plot_cm)
        self._build_results_df()

        print("\n✅  Pipeline selesai. Panggil .show_comparison() untuk hasil.")

    def show_comparison(self, top_n: int = 10):
        """Tampilkan tabel komparasi Baseline vs Feature Selection."""
        if self.results_df is None:
            print("⚠  Belum ada hasil. Jalankan .run() terlebih dahulu.")
            return

        df = self.results_df.copy()

        print("\n" + "═"*90)
        print("  KOMPARASI BASELINE vs FEATURE SELECTION")
        print("═"*90)

        # ── Per model ────────────────────────────────────────────────────────
        for model_name in self.DEFAULT_MODELS:
            print(f"\n  ▸ Model: {model_name}")
            print(f"  {'─'*80}")

            baseline_row = df[
                (df["Stage"] == "Baseline") & (df["Model"] == model_name)
            ]
            fs_rows = df[
                (df["Stage"] == "FeatureSelection") & (df["Model"] == model_name)
            ].sort_values("F1 Macro", ascending=False)

            if baseline_row.empty:
                print("    (tidak ada hasil baseline)")
                continue

            b = baseline_row.iloc[0]
            print(
                f"  {'Stage':<22} {'N Fitur':<10} {'Accuracy':<12} "
                f"{'F1 Macro':<12} {'F1 Weighted':<14} {'Recall Mac':<12} "
                f"{'Prec Mac':<12} {'Train(s)':<10} {'Test(s)'}"
            )
            print(f"  {'─'*80}")
            self._print_row(b)

            best_fs = fs_rows.iloc[0] if not fs_rows.empty else None
            for _, row in fs_rows.head(top_n).iterrows():
                marker = " ◀ TERBAIK" if best_fs is not None and row.name == best_fs.name else ""
                self._print_row(row, marker=marker)

        # ── Global best ───────────────────────────────────────────────────────
        print("\n" + "═"*90)
        print("  GLOBAL BEST (F1 Macro)")
        print("═"*90)
        best = df.sort_values("F1 Macro", ascending=False).iloc[0]
        self._print_row(best, prefix="  🏆 ")

        # ── Delta table ───────────────────────────────────────────────────────
        print("\n" + "═"*90)
        print("  DELTA: FS_TERBAIK − BASELINE (per Model)")
        print("═"*90)
        self._show_delta_table(df)

        print("═"*90)

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE: STEP SHAP
    # ═════════════════════════════════════════════════════════════════════════

    def _step_shap(self, plot: bool = True):
        print("\n[1/3] Menghitung SHAP Feature Importance (dari X_train) ...")

        model_map = {
            "xgboost":      XGBClassifier(n_estimators=100, max_depth=5,
                                          random_state=42),
            "lgbm":         LGBMClassifier(n_estimators=100, random_state=42,
                                           verbose=-1),
            "randomforest": RandomForestClassifier(n_estimators=100,
                                                   random_state=42),
        }
        shap_clf = model_map.get(self.shap_model, model_map["xgboost"])

        with tqdm(total=3, desc="  SHAP", ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            shap_clf.fit(self.X_train, self.y_train)
            pbar.update(1)

            explainer   = shap.TreeExplainer(shap_clf)
            shap_values = explainer.shap_values(self.X_train)   # ← X_train ✅
            pbar.update(1)

            if isinstance(shap_values, list):
                # multi-class list of arrays → stack → (n_samples, n_features, n_classes)
                sv = np.stack(shap_values, axis=2)
            else:
                sv = shap_values

            if sv.ndim == 3:
                mean_abs = np.abs(sv).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(sv).mean(axis=0)

            self.shap_importance_df = (
                pd.DataFrame({
                    "Feature":         self.X_train.columns,
                    "SHAP_Importance": mean_abs,
                })
                .sort_values("SHAP_Importance", ascending=False)
                .reset_index(drop=True)
            )
            pbar.update(1)

        print(f"\n  Top-10 Fitur (SHAP dari X_train — model: {self.shap_model}):")
        print(self.shap_importance_df.head(10).to_string(index=False))

        if plot:
            plt.figure(figsize=(10, 6))
            shap_top = self.shap_importance_df.head(20)
            plt.barh(
                shap_top["Feature"][::-1].values,
                shap_top["SHAP_Importance"][::-1].values,
                color="#4C72B0",
            )
            plt.xlabel("Mean |SHAP value|")
            plt.title(f"Top-20 SHAP Feature Importance ({self.shap_model.upper()}, X_train)")
            plt.tight_layout()
            plt.show()

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE: STEP BASELINE
    # ═════════════════════════════════════════════════════════════════════════

    def _step_baseline(self):
        print("\n[2/3] Melatih Baseline (semua fitur, tanpa SMOTE/normalisasi) ...")

        scaler         = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled  = scaler.transform(self.X_test)

        models   = list(self.DEFAULT_MODELS.items())
        n_models = len(models)

        with tqdm(total=n_models, desc="  Baseline", ncols=70,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for model_name, clf in models:
                clf_fit = clf.__class__(**clf.get_params())

                t0 = time.time()
                clf_fit.fit(X_train_scaled, self.y_train)
                train_t = time.time() - t0

                t0 = time.time()
                y_pred = clf_fit.predict(X_test_scaled)
                test_t = time.time() - t0

                self.baseline_results[model_name] = self._collect_metrics(
                    stage="Baseline",
                    model=model_name,
                    n_features=self.X_train.shape[1],
                    percentile=100,
                    y_true=self.y_test,
                    y_pred=y_pred,
                    train_time=train_t,
                    test_time=test_t,
                )
                pbar.update(1)

        print("  Baseline selesai.")

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE: STEP FEATURE SELECTION
    # ═════════════════════════════════════════════════════════════════════════

    def _step_feature_selection(self, plot_cm: bool = True):
        print("\n[3/3] Feature Selection + SMOTETomek ...")

        sampler    = SMOTETomek(random_state=42)
        n_total    = len(self.shap_importance_df)
        models     = list(self.DEFAULT_MODELS.items())
        total_iter = len(self.percentiles) * len(models)

        with tqdm(total=total_iter, desc="  FS Loop", ncols=70,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:

            for P in self.percentiles:
                N_feat    = int(np.ceil(n_total * P / 100))
                sel_feats = self.shap_importance_df.head(N_feat)["Feature"].tolist()

                X_tr_p = self.X_train[sel_feats]
                X_te_p = self.X_test[sel_feats]

                X_res, y_res = sampler.fit_resample(X_tr_p, self.y_train)

                scaler         = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_res)
                X_test_scaled  = scaler.transform(X_te_p)

                for model_name, clf in models:
                    clf_fit = clf.__class__(**clf.get_params())

                    t0 = time.time()
                    clf_fit.fit(X_train_scaled, y_res)
                    train_t = time.time() - t0

                    t0 = time.time()
                    y_pred = clf_fit.predict(X_test_scaled)
                    test_t = time.time() - t0

                    key = f"P{P}_{model_name}"
                    self.fs_results[key] = self._collect_metrics(
                        stage="FeatureSelection",
                        model=model_name,
                        n_features=N_feat,
                        percentile=P,
                        y_true=self.y_test,
                        y_pred=y_pred,
                        train_time=train_t,
                        test_time=test_t,
                    )

                    if plot_cm:
                        self._plot_cm(self.y_test, y_pred, model_name, P)

                    pbar.update(1)

        print("  Feature Selection selesai.")

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE: BUILD RESULTS DF
    # ═════════════════════════════════════════════════════════════════════════

    def _build_results_df(self):
        rows = list(self.baseline_results.values()) + list(self.fs_results.values())
        self.results_df = pd.DataFrame(rows)

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE: HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _collect_metrics(stage, model, n_features, percentile,
                         y_true, y_pred, train_time, test_time):
        return {
            "Stage":              stage,
            "Model":              model,
            "N Fitur":            n_features,
            "Persentil (%)":      percentile,
            "Accuracy":           round(accuracy_score(y_true, y_pred), 4),
            "F1 Macro":           round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
            "F1 Weighted":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Recall Macro":       round(recall_score(y_true, y_pred, average="macro",    zero_division=0), 4),
            "Recall Weighted":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Precision Macro":    round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
            "Precision Weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Train Time (s)":     round(train_time, 4),
            "Test Time (s)":      round(test_time, 4),
        }

    def _plot_cm(self, y_true, y_pred, model_name, percentile):
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        labels  = [str(c) for c in self.class_labels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Confusion Matrix — {model_name} | P={percentile}%",
            fontsize=13, fontweight="bold",
        )
        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2f"],
            ["Raw Count", "Normalized (per True Class)"],
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, vmin=0 if fmt == ".2f" else None,
                vmax=1 if fmt == ".2f" else None,
            )
            ax.set_title(title)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _print_row(row, marker: str = "", prefix: str = "  "):
        print(
            f"{prefix}"
            f"{str(row['Stage']):<22} "
            f"{str(int(row['N Fitur'])):<10} "
            f"{float(row['Accuracy']):<12.4f}"
            f"{float(row['F1 Macro']):<12.4f}"
            f"{float(row['F1 Weighted']):<14.4f}"
            f"{float(row['Recall Macro']):<12.4f}"
            f"{float(row['Precision Macro']):<12.4f}"
            f"{float(row['Train Time (s)']):<10.4f}"
            f"{float(row['Test Time (s)']):.4f}"
            f"{marker}"
        )

    def _show_delta_table(self, df: pd.DataFrame):
        metrics = ["Accuracy", "F1 Macro", "F1 Weighted",
                   "Recall Macro", "Precision Macro"]
        header = f"  {'Model':<16}" + "".join(f"{m:<16}" for m in metrics)
        print(header)
        print(f"  {'─'*80}")

        for model_name in self.DEFAULT_MODELS:
            b_row = df[(df["Stage"] == "Baseline") & (df["Model"] == model_name)]
            fs_rows = df[(df["Stage"] == "FeatureSelection") & (df["Model"] == model_name)]

            if b_row.empty or fs_rows.empty:
                continue

            b_vals  = b_row.iloc[0]
            fs_best = fs_rows.sort_values("F1 Macro", ascending=False).iloc[0]

            deltas = [fs_best[m] - b_vals[m] for m in metrics]
            delta_str = "".join(
                f"{'▲' if d > 0 else '▼' if d < 0 else '='}{abs(d):<14.4f}"
                for d in deltas
            )
            print(f"  {model_name:<16}{delta_str}")


# ── Contoh penggunaan ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ganti dengan X_train, X_test, y_train, y_test Anda
    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import train_test_split
    # X, y = make_classification(n_samples=2000, n_features=30, n_classes=4,
    #                            n_informative=15, random_state=42)
    # X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    # y = pd.Series(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    #                                                      stratify=y, random_state=42)

    # pipe = MLPipeline(X_train, X_test, y_train, y_test)
    # pipe.run()
    # pipe.show_comparison()
    print("Import class MLPipeline, lalu jalankan contoh di atas.")
