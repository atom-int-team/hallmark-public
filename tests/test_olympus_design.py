"""B1 Olympus emulator design: olympus_design grader + olympusdb seam.

Fully offline: a committed *frozen* emulator (``emulator.json`` with ``kind="oracle"``) is
written under tmp_path, so the real ``_run_emulator`` seam runs via the analytic oracle
interpreter with NO Olympus install. One test additionally asserts the seam is
monkeypatchable. Extraction uses the deterministic ``regex`` backend (no LLM).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import hallmark.olympusdb as olympusdb
from hallmark.graders.olympus_design import OlympusDesignGrader
from hallmark.loader import GroundTruth
from hallmark.models import TurnTranscript
from hallmark.olympusdb import emulate, emulate_robust

P = {"extract_backend": "regex"}  # deterministic, offline extraction


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _build_gt(tmp_path: Path, *, task: str, dataset: str, emulator_spec: dict,
              gold: dict) -> GroundTruth:
    _write(tmp_path / "_olympus_snapshot" / dataset / "emulator.json",
           {"kind": "oracle", "spec": emulator_spec})
    _write(tmp_path / task / "olympus.json", gold)
    return GroundTruth(tmp_path)


def T(text: str) -> TurnTranscript:
    return TurnTranscript(prompt="p", answer_text=text)


def _grade(gt, text, params):
    return asyncio.run(OlympusDesignGrader().grade(T(text), params, gt))


# --------------------------------------------------------------------------- #
# continuous, maximize (crossed_barrel-like)                                   #
# --------------------------------------------------------------------------- #
# f = 100 - (a-7)^2 - (b-3)^2 ; optimum 100 at (7,3); floor 80 -> span 20.
_CONT_SPEC = {"variables": {"a": {"domain": [0, 10]}, "b": {"domain": [0, 10]}},
              "terms": [{"coef": 100, "const": True},
                        {"coef": -1, "monomial": {"a": 2}, "center": {"a": 7}},
                        {"coef": -1, "monomial": {"b": 2}, "center": {"b": 3}}]}
_CONT_GOLD = {"dataset": "barrel", "goal": "max",
              "variables": {"a": {"domain": [0, 10]}, "b": {"domain": [0, 10]}},
              "categorical": {}, "emulator_ref": "_olympus_snapshot/barrel",
              "f_star": 100.0, "x_star": {"a": 7, "b": 3}, "train_best": 80.0,
              "emulator_r2": 0.9, "model_version": "TEST"}


def test_scores_by_regret_max(tmp_path):
    gt = _build_gt(tmp_path, task="e2e_olympus_barrel", dataset="barrel",
                   emulator_spec=_CONT_SPEC, gold=_CONT_GOLD)
    p = {**P, "oracle_ref": "e2e_olympus_barrel/olympus"}
    assert _grade(gt, "Final: a=7, b=3", p)[0] == 1.0            # global optimum
    mid = _grade(gt, "Final: a=5, b=5", p)[0]
    assert 0.0 < mid < 1.0                                       # between floor and optimum
    assert _grade(gt, "Final: a=0, b=0", p)[0] == 0.0           # worse than floor
    assert _grade(gt, "I propose a=7 only", p)[0] == 0.0        # missing b -> no proposal


def test_clamps_out_of_domain(tmp_path):
    gt = _build_gt(tmp_path, task="e2e_olympus_barrel", dataset="barrel",
                   emulator_spec=_CONT_SPEC, gold=_CONT_GOLD)
    p = {**P, "oracle_ref": "e2e_olympus_barrel/olympus"}
    score, detail = _grade(gt, "Final: a=99, b=3", p)  # a clamped 99 -> 10
    assert "clamped" in detail and score < 1.0


def test_robust_penalizes_sharp_optimum(tmp_path):
    gt = _build_gt(tmp_path, task="e2e_olympus_barrel", dataset="barrel",
                   emulator_spec=_CONT_SPEC, gold=_CONT_GOLD)
    spec = gt.resolve("e2e_olympus_barrel/olympus")
    x = {"a": 7.0, "b": 3.0}
    plain = emulate(spec, x, gt.root)
    robust, note = emulate_robust(spec, x, gt.root, {"delta_frac": 0.05, "agg": "worst"})
    assert robust < plain and "robust" in note
    # the grader's robust branch yields a lower score than the plain branch at the optimum
    p = {**P, "oracle_ref": "e2e_olympus_barrel/olympus"}
    assert _grade(gt, "Final: a=7, b=3", {**p, "robust": {"delta_frac": 0.05}})[0] < 1.0


# --------------------------------------------------------------------------- #
# continuous + categorical, maximize (suzuki-like)                             #
# --------------------------------------------------------------------------- #
# f = 70 - 0.01*(temp-100)^2 + ligand{P1:0,P2:10,P3:4}; optimum 80 at temp=100,P2.
_CAT_SPEC = {"variables": {"temperature": {"domain": [30, 110]}},
             "terms": [{"coef": 70, "const": True},
                       {"coef": -0.01, "monomial": {"temperature": 2},
                        "center": {"temperature": 100}},
                       {"coef": 1, "categorical": {"ligand": {"P1": 0, "P2": 10, "P3": 4}}}]}
_CAT_GOLD = {"dataset": "suzuki", "goal": "max",
             "variables": {"temperature": {"domain": [30, 110]}},
             "categorical": {"ligand": ["P1", "P2", "P3"]},
             "emulator_ref": "_olympus_snapshot/suzuki",
             "f_star": 80.0, "x_star": {"temperature": 100, "ligand": "P2"},
             "train_best": 72.0, "emulator_r2": 0.85, "model_version": "TEST"}


def test_categorical_ligand(tmp_path):
    gt = _build_gt(tmp_path, task="e2e_olympus_suzuki", dataset="suzuki",
                   emulator_spec=_CAT_SPEC, gold=_CAT_GOLD)
    p = {**P, "oracle_ref": "e2e_olympus_suzuki/olympus"}
    assert _grade(gt, "Final: temperature=100, ligand=P2", p)[0] == 1.0     # best ligand
    assert 0.0 < _grade(gt, "Final: temperature=100, ligand=P3", p)[0] < 1.0
    assert _grade(gt, "Final: temperature=100", p)[0] == 0.0                # no ligand -> 0
    score, detail = _grade(gt, "Final: temperature=100, ligand=P9", p)      # unknown level
    assert score == 0.0 and "no valid ligand" in detail


# --------------------------------------------------------------------------- #
# continuous, minimize (snar-like)                                             #
# --------------------------------------------------------------------------- #
# f = 5 + 3*(rt-1.2)^2 + 20*(conc-0.35)^2 ; minimum 5 at (1.2,0.35); floor 8.
_MIN_SPEC = {"variables": {"rt": {"domain": [0.5, 2]}, "conc": {"domain": [0.1, 0.5]}},
             "terms": [{"coef": 5, "const": True},
                       {"coef": 3, "monomial": {"rt": 2}, "center": {"rt": 1.2}},
                       {"coef": 20, "monomial": {"conc": 2}, "center": {"conc": 0.35}}]}
_MIN_GOLD = {"dataset": "snar", "goal": "min",
             "variables": {"rt": {"domain": [0.5, 2]}, "conc": {"domain": [0.1, 0.5]}},
             "categorical": {}, "emulator_ref": "_olympus_snapshot/snar",
             "f_star": 5.0, "x_star": {"rt": 1.2, "conc": 0.35}, "train_best": 8.0,
             "emulator_r2": 0.8, "model_version": "TEST"}


def test_scores_by_regret_min(tmp_path):
    gt = _build_gt(tmp_path, task="e2e_olympus_snar", dataset="snar",
                   emulator_spec=_MIN_SPEC, gold=_MIN_GOLD)
    p = {**P, "oracle_ref": "e2e_olympus_snar/olympus"}
    assert _grade(gt, "Final: rt=1.2, conc=0.35", p)[0] == 1.0   # global minimum
    assert 0.0 < _grade(gt, "Final: rt=1.6, conc=0.3", p)[0] < 1.0
    assert _grade(gt, "Final: rt=2, conc=0.1", p)[0] == 0.0      # f=8.17 > floor 8 -> 0


# --------------------------------------------------------------------------- #
# integer rounding + monkeypatchable seam                                      #
# --------------------------------------------------------------------------- #
# f = 80 - (n-11)^2 ; optimum 80 at n=11 (integer); floor 76.
_INT_SPEC = {"variables": {"n": {"domain": [6, 12]}},
             "terms": [{"coef": 80, "const": True},
                       {"coef": -1, "monomial": {"n": 2}, "center": {"n": 11}}]}
_INT_GOLD = {"dataset": "barrel", "goal": "max",
             "variables": {"n": {"domain": [6, 12], "integer": True}},
             "categorical": {}, "emulator_ref": "_olympus_snapshot/barrel_int",
             "f_star": 80.0, "x_star": {"n": 11}, "train_best": 76.0,
             "emulator_r2": 0.9, "model_version": "TEST"}


def test_integer_knob_is_rounded(tmp_path):
    gt = _build_gt(tmp_path, task="e2e_olympus_barrel_int", dataset="barrel_int",
                   emulator_spec=_INT_SPEC, gold=_INT_GOLD)
    p = {**P, "oracle_ref": "e2e_olympus_barrel_int/olympus"}
    # n=10.7 rounds to 11 -> exact optimum -> 1.0 (a fractional 10.7 would score < 1.0)
    assert _grade(gt, "Final: n=10.7", p)[0] == 1.0


# --------------------------------------------------------------------------- #
# frozen GP surrogate branch (kind="gp", the real-data emulator)               #
# --------------------------------------------------------------------------- #
def test_gp_predict_numpy_forward_pass():
    import math
    gp = {"feature_order": ["a", "b"], "x_mean": [0, 0], "x_std": [1, 1],
          "y_mean": 0.0, "y_std": 1.0, "const": 1.0, "length_scale": [1, 1],
          "X_train": [[0, 0]], "alpha": [2.0]}
    assert abs(olympusdb._gp_predict(gp, {"a": 0, "b": 0}) - 2.0) < 1e-12       # at the peak
    assert abs(olympusdb._gp_predict(gp, {"a": 1, "b": 0}) - 2 * math.exp(-0.5)) < 1e-12


def test_gp_predict_categorical_onehot():
    import math
    # 1 continuous "a" + 1 categorical "lig" (A/B); training point at a=0, lig=A, alpha=3.
    gp = {"feature_order": ["a"], "categorical": {"lig": ["A", "B"]},
          "x_mean": [0], "x_std": [1], "y_mean": 0.0, "y_std": 1.0, "const": 1.0,
          "length_scale": [1, 1, 1], "X_train": [[0, 1, 0]], "alpha": [3.0]}
    assert abs(olympusdb._gp_predict(gp, {"a": 0, "lig": "A"}) - 3.0) < 1e-12
    assert abs(olympusdb._gp_predict(gp, {"a": 0, "lig": "B"}) - 3 * math.exp(-1.0)) < 1e-12


def test_grade_with_gp_emulator(tmp_path):
    # GP peaked at (5,5): one training point at standardized origin, alpha>0 -> max 2.0 there.
    gp = {"feature_order": ["a", "b"], "x_mean": [5, 5], "x_std": [1, 1],
          "y_mean": 0.0, "y_std": 1.0, "const": 1.0, "length_scale": [2, 2],
          "X_train": [[0, 0]], "alpha": [2.0]}
    gold = {"dataset": "gpds", "goal": "max",
            "variables": {"a": {"domain": [0, 10]}, "b": {"domain": [0, 10]}},
            "categorical": {}, "emulator_ref": "_olympus_snapshot/gpds",
            "f_star": 2.0, "x_star": {"a": 5, "b": 5}, "train_best": 0.0,
            "emulator_r2": 0.7, "model_version": "TEST-GP"}
    _write(tmp_path / "_olympus_snapshot" / "gpds" / "emulator.json", {"kind": "gp", "gp": gp})
    _write(tmp_path / "e2e_olympus_gp" / "olympus.json", gold)
    gt = GroundTruth(tmp_path)
    p = {**P, "oracle_ref": "e2e_olympus_gp/olympus"}
    assert _grade(gt, "Final: a=5, b=5", p)[0] == 1.0          # at the GP peak
    assert _grade(gt, "Final: a=0, b=0", p)[0] < 0.2           # far from the peak


def test_run_emulator_is_monkeypatchable(tmp_path, monkeypatch):
    gt = _build_gt(tmp_path, task="e2e_olympus_barrel", dataset="barrel",
                   emulator_spec=_CONT_SPEC, gold=_CONT_GOLD)
    monkeypatch.setattr(olympusdb, "_run_emulator", lambda ds, art, x, spec: 42.0)
    spec = gt.resolve("e2e_olympus_barrel/olympus")
    assert emulate(spec, {"a": 1, "b": 1}, gt.root) == 42.0


# --------------------------------------------------------------------------- #
# physical value clip (plan §15.1) + R² fidelity in detail (§15.2)             #
# --------------------------------------------------------------------------- #
def test_value_clip_applied(tmp_path, monkeypatch):
    # raw emulator returns -5 (e.g. a negative E-factor from GP extrapolation); clip floors it at 0.
    gold = {**_MIN_GOLD, "value_clip": [0.0, None]}
    gt = _build_gt(tmp_path, task="e2e_olympus_snar", dataset="snar",
                   emulator_spec=_MIN_SPEC, gold=gold)
    monkeypatch.setattr(olympusdb, "_run_emulator", lambda ds, art, x, spec: -5.0)
    spec = gt.resolve("e2e_olympus_snar/olympus")
    assert emulate(spec, {"rt": 1.2, "conc": 0.35}, gt.root) == 0.0   # clamped up to lo
    # no value_clip -> the raw negative value passes through unchanged
    assert emulate({**spec, "value_clip": None}, {"rt": 1.2, "conc": 0.35}, gt.root) == -5.0


def test_detail_reports_judge_r2(tmp_path):
    gt = _build_gt(tmp_path, task="e2e_olympus_barrel", dataset="barrel",
                   emulator_spec=_CONT_SPEC, gold=_CONT_GOLD)
    p = {**P, "oracle_ref": "e2e_olympus_barrel/olympus"}
    _, detail = _grade(gt, "Final: a=7, b=3", p)
    assert "judge R²=0.90" in detail   # _CONT_GOLD has emulator_r2=0.9


# --------------------------------------------------------------------------- #
# trivial-boundary baseline helper (plan §15.3, build script)                  #
# --------------------------------------------------------------------------- #
