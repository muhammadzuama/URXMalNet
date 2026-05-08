"""
XMalNet
=======
Automated pipeline for malware classification:
  1. Train Random Forest, XGBoost, and LightGBM models
  2. Perform SHAP analysis using the trained XGBoost model
  3. Apply SHAP-based feature selection using quartile thresholds:
       - 25th percentile (Q1)
       - 50th percentile / median (Q2)
       - 75th percentile (Q3)
  4. Perform SMOTE oversampling (4 variants) × 3 quartile-based feature subsets → select the best combination
  5. Save the summary results to CSV

Usage
-----
    from xmalnet import XMalNet

    xmn = XMalNet(class_labels=[0, 1, 2, 3])   # optional
    xmn.fit(X_train, X_test, y_train, y_test)
    xmn.summary()                               # comparison table
    xmn.best_result()                           # best combination
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
class XMalNet:
    """
    Parameters
    ----------
    class_labels : list, optional
        Label kelas untuk laporan klasifikasi.
        Jika None, akan diambil otomatis dari y_train saat fit().
    percentiles : list[int], optional
        Persentil SHAP untuk feature selection. Default: [25, 50, 75].
    save_shap_plot : bool, optional
        Simpan SHAP summary plot ke file PNG. Default: True.
    shap_plot_path : str, optional
        Path file SHAP plot. Default: "shap_xgb_summary.png".
    save_results_csv : str or None, optional
        Jika diberikan, simpan tabel hasil ke file CSV ini. Default: None.
    verbose : bool, optional
        Tampilkan progress bar saat training. Default: True.
        Set False untuk menyembunyikan semua output saat fit().
    """

    # ── RF / XGB / LGBM default params ───────────────────────────────────────
    _RF_PARAMS = dict(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
    )
    _XGB_PARAMS = dict(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.2,
        random_state=42,
    )
    _LGBM_PARAMS = dict(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.2,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )

    def __init__(
        self,
        class_labels=None,
        percentiles=None,
        save_shap_plot=True,
        shap_plot_path="shap_xgb_summary.png",
        save_results_csv=None,
        verbose=True,
    ):
        self.class_labels    = class_labels
        self.percentiles     = percentiles or [25, 50, 75]
        self.save_shap_plot  = save_shap_plot
        self.shap_plot_path  = shap_plot_path
        self.save_results_csv = save_results_csv
        self.verbose         = verbose

        # ── internal state ────────────────────────────────────────────────────
        self.rf_model_            = None
        self.xgb_model_           = None
        self.lgbm_model_          = None
        self.shap_importance_df_  = None
        self.results_df_          = None
        self.best_model_          = None
        self.best_features_       = None
        self._baseline_metrics_   = {}   # simpan metrik baseline
        self._is_fitted           = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X_train, X_test, y_train, y_test):
        """
        Jalankan seluruh pipeline XMalNet.

        Parameters
        ----------
        X_train, X_test : pd.DataFrame
        y_train, y_test : array-like
        """
        X_train = pd.DataFrame(X_train)
        X_test  = pd.DataFrame(X_test)
        y_train = np.array(y_train)
        y_test  = np.array(y_test)

        if self.class_labels is None:
            self.class_labels = sorted(np.unique(y_train))

        # Total langkah: 3 baseline + 1 SHAP + (P × SMOTE) grid + 1 retrain
        smote_count      = len(self._smote_variants())
        grid_steps       = len(self.percentiles) * smote_count
        total_steps      = 3 + 1 + grid_steps + 1

        pbar = self._make_pbar(total_steps, "XMalNet Training")

        # ── Step 1: Baseline Models ───────────────────────────────────────────
        self._set_desc(pbar, "Training Random Forest")
        self.rf_model_ = self._train_silent(
            "Random Forest",
            RandomForestClassifier(**self._RF_PARAMS),
            X_train, X_test, y_train, y_test,
        )
        self._step(pbar)

        self._set_desc(pbar, "Training XGBoost")
        self.xgb_model_ = self._train_silent(
            "XGBoost",
            XGBClassifier(**self._XGB_PARAMS),
            X_train, X_test, y_train, y_test,
        )
        self._step(pbar)

        self._set_desc(pbar, "Training LightGBM")
        self.lgbm_model_ = self._train_silent(
            "LightGBM",
            LGBMClassifier(**self._LGBM_PARAMS),
            X_train, X_test, y_train, y_test,
        )
        self._step(pbar)

        # ── Step 2: SHAP ──────────────────────────────────────────────────────
        self._set_desc(pbar, "SHAP Analysis (XGBoost)")
        self.shap_importance_df_ = self._compute_shap_silent(self.xgb_model_, X_test)
        self._step(pbar)

        # ── Step 3: SMOTE Grid ────────────────────────────────────────────────
        results_all = {}
        smote_variants = self._smote_variants()

        for P in self.percentiles:
            N        = int(np.ceil(len(self.shap_importance_df_) * P / 100))
            selected = self.shap_importance_df_.head(N)["Feature"].tolist()
            X_tr_p   = X_train[selected]
            X_te_p   = X_test[selected]

            for smote_name, sampler in smote_variants.items():
                self._set_desc(pbar, f"SMOTE Grid P={P}% | {smote_name}")

                X_res, y_res = sampler.fit_resample(X_tr_p, y_train)
                model = XGBClassifier(**self.xgb_model_.get_params())
                model.fit(X_res, y_res)

                y_pred = model.predict(X_te_p)
                report = classification_report(y_test, y_pred, output_dict=True)

                key = f"P{P}_{smote_name}"
                results_all[key] = {
                    "Persentil"  : P,
                    "N Fitur"    : N,
                    "SMOTE"      : smote_name,
                    "Accuracy"   : report["accuracy"],
                    "F1 Macro"   : report["macro avg"]["f1-score"],
                    "F1 Weighted": report["weighted avg"]["f1-score"],
                    "Precision"  : report["macro avg"]["precision"],
                    "Recall"     : report["macro avg"]["recall"],
                }
                self._step(pbar)

        self.results_df_ = pd.DataFrame(results_all).T.round(4)

        # ── Step 4: Retrain Best ──────────────────────────────────────────────
        best_row     = self.results_df_.sort_values("F1 Macro", ascending=False).iloc[0]
        N            = int(best_row["N Fitur"])
        smote_name   = best_row["SMOTE"]

        self._set_desc(pbar, f"Retraining Best Model [{smote_name}, P{int(best_row['Persentil'])}%]")

        self.best_features_ = self.shap_importance_df_.head(N)["Feature"].tolist()
        X_tr_best = X_train[self.best_features_]
        X_te_best = X_test[self.best_features_]

        X_res, y_res = smote_variants[smote_name].fit_resample(X_tr_best, y_train)
        self.best_model_ = XGBClassifier(**self.xgb_model_.get_params())
        self.best_model_.fit(X_res, y_res)

        # simpan metrik best model
        y_pred_best = self.best_model_.predict(X_te_best)
        self._best_metrics_ = self._calc_metrics(y_test, y_pred_best)

        self._step(pbar)
        self._close_pbar(pbar)

        self._is_fitted = True

        if self.save_results_csv:
            self.results_df_.to_csv(self.save_results_csv)
            if self.verbose:
                print(f"\n💾  Hasil disimpan ke: {self.save_results_csv}")

        # ── Print ringkasan akhir ──────────────────────────────────────────────
        self._print_final_summary(y_test, y_pred_best, best_row)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X):
        """Prediksi dengan model terbaik hasil fit()."""
        self._check_fitted()
        return self.best_model_.predict(pd.DataFrame(X)[self.best_features_])

    def predict_proba(self, X):
        """Probabilitas prediksi dengan model terbaik."""
        self._check_fitted()
        return self.best_model_.predict_proba(pd.DataFrame(X)[self.best_features_])

    # ── Report methods ────────────────────────────────────────────────────────

    def summary(self):
        """Tampilkan tabel perbandingan semua kombinasi P × SMOTE."""
        self._check_fitted()
        df = self.results_df_.sort_values("F1 Macro", ascending=False)
        print("\n" + "─"*70)
        print("  📊  Perbandingan Semua Kombinasi (Persentil × SMOTE)")
        print("─"*70)
        print(df.to_string())
        print("─"*70)
        return df

    def best_result(self):
        """Tampilkan kombinasi terbaik."""
        self._check_fitted()
        best = self.results_df_.sort_values("F1 Macro", ascending=False).iloc[0]
        print("\n" + "─"*50)
        print("  ✅  Kombinasi Terbaik")
        print("─"*50)
        for col, val in best.items():
            print(f"   {col:<16}: {val}")
        print("─"*50)
        return best

    def shap_importance(self, top_n=20):
        """Tampilkan top-N fitur berdasarkan SHAP importance."""
        self._check_fitted()
        df = self.shap_importance_df_.head(top_n)
        print(f"\n  🔍  Top {top_n} Features (XGBoost - SHAP):")
        print(df.to_string(index=False))
        return df

    def baseline_summary(self):
        """Tampilkan perbandingan metrik semua model baseline."""
        self._check_fitted()
        df = pd.DataFrame(self._baseline_metrics_).T.round(4)
        print("\n" + "─"*60)
        print("  📋  Baseline Model Comparison")
        print("─"*60)
        print(df.to_string())
        print("─"*60)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _train_silent(self, name, model, X_train, X_test, y_train, y_test):
        """Latih model tanpa print apapun, simpan metrik internal."""
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t0

        self._baseline_metrics_[name] = {
            **self._calc_metrics(y_test, y_pred),
            "Train Time (s)": round(train_time, 4),
            "Test Time (s)" : round(test_time, 4),
        }
        return model

    def _compute_shap_silent(self, model, X_test):
        """Hitung SHAP tanpa output teks, simpan plot jika diminta."""
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure()
        shap.summary_plot(
            shap_values, X_test,
            class_names=[str(c) for c in self.class_labels],
            show=False,
        )
        plt.title("SHAP Summary Plot - XGBoost")
        plt.tight_layout()
        if self.save_shap_plot:
            plt.savefig(self.shap_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        if shap_values.ndim == 3:
            shap_mean = np.abs(shap_values).mean(axis=(0, 2))
        else:
            shap_mean = np.abs(shap_values).mean(axis=0)

        return (
            pd.DataFrame({"Feature": X_test.columns, "SHAP_Importance": shap_mean})
            .sort_values("SHAP_Importance", ascending=False)
            .reset_index(drop=True)
        )

    def _calc_metrics(self, y_true, y_pred):
        return {
            "Accuracy"        : round(accuracy_score(y_true, y_pred), 4),
            "Precision (W)"   : round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Precision (M)"   : round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
            "Recall (W)"      : round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Recall (M)"      : round(recall_score(y_true, y_pred, average="macro",    zero_division=0), 4),
            "F1 Weighted"     : round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "F1 Macro"        : round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        }

    def _print_final_summary(self, y_test, y_pred_best, best_row):
        """Print ringkasan hasil setelah fit() selesai."""
        print("\n" + "═"*60)
        print("  XMalNet — Training Complete ✓")
        print("═"*60)

        # Baseline comparison
        print("\n  📋  Baseline Models:")
        print(f"  {'Model':<16} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}")
        print("  " + "─"*50)
        for name, m in self._baseline_metrics_.items():
            print(f"  {name:<16} {m['Accuracy']:>10} {m['F1 Macro']:>10} {m['F1 Weighted']:>12}")

        # Best SMOTE config
        print(f"\n  🏆  Best SMOTE Config  : P={int(best_row['Persentil'])}% | {best_row['SMOTE']} | {int(best_row['N Fitur'])} features")

        # Best model metrics
        m = self._best_metrics_
        print(f"\n  🎯  Best Model Metrics :")
        print(f"      Accuracy      : {m['Accuracy']}")
        print(f"      F1 Macro      : {m['F1 Macro']}")
        print(f"      F1 Weighted   : {m['F1 Weighted']}")
        print(f"      Precision (M) : {m['Precision (M)']}")
        print(f"      Recall (M)    : {m['Recall (M)']}")

        # Classification report
        print("\n  📄  Classification Report (Best Model):\n")
        print(classification_report(
            y_test, y_pred_best,
            target_names=[str(x) for x in self.class_labels],
            zero_division=0,
        ))

        if self.save_shap_plot:
            print(f"  🖼   SHAP plot disimpan ke : {self.shap_plot_path}")
        if self.save_results_csv:
            print(f"  💾  CSV disimpan ke       : {self.save_results_csv}")

        print("═"*60 + "\n")

    # ── Progress bar helpers ──────────────────────────────────────────────────

    def _make_pbar(self, total, desc):
        if not self.verbose:
            return None
        if _TQDM_AVAILABLE:
            return tqdm(
                total=total,
                desc=desc,
                unit="step",
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        # fallback jika tqdm tidak tersedia
        return {"n": 0, "total": total, "desc": desc}

    def _step(self, pbar):
        if pbar is None:
            return
        if _TQDM_AVAILABLE:
            pbar.update(1)
        else:
            pbar["n"] += 1
            pct = int(pbar["n"] / pbar["total"] * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {pbar['n']}/{pbar['total']} — {pbar['desc']}   ", end="", flush=True)

    def _set_desc(self, pbar, desc):
        if pbar is None:
            return
        if _TQDM_AVAILABLE:
            pbar.set_description(desc)
        else:
            pbar["desc"] = desc

    def _close_pbar(self, pbar):
        if pbar is None:
            return
        if _TQDM_AVAILABLE:
            pbar.close()
        else:
            print()  # newline setelah progress bar fallback

    # ── Misc ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _smote_variants():
        return {
            "SMOTE"          : SMOTE(random_state=42),
            "SMOTEENN"       : SMOTEENN(random_state=42),
            "SMOTETomek"     : SMOTETomek(random_state=42),
            "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
        }

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("XMalNet belum di-fit. Panggil .fit() terlebih dahulu.")