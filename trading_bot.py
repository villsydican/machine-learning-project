{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "include_colab_link": true
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/villsydican/machine-learning-project/blob/main/trading_bot.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "O1WOFeXX1CW0",
        "outputId": "c9fd5170-b942-4830-bf44-1037da8deadc"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "#CONNECTING THE MODEL TO DRIVE FOR EASY DATA FETCHING AND MODEL SAVING\n",
        "\n",
        "# CELL 1 - SETTINGS (run this first; it also connects Google Drive)\n",
        "import os, glob, json\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import joblib\n",
        "\n",
        "try:\n",
        "    from google.colab import drive\n",
        "    drive.mount(\"/content/drive\")\n",
        "except Exception:\n",
        "    pass\n",
        "\n",
        "DATA_DIR   = \"/content/drive/MyDrive/fx_data\"                  # same CSV folder as before\n",
        "MODEL_PATH = \"/content/drive/MyDrive/indicator_model.joblib\"   # where the new model is saved\n",
        "\n",
        "CLASS_NAMES = [\"buy\", \"hold\", \"sell\"]   # 0 = buy, 1 = hold, 2 = sell (same as the CNN)\n",
        "ATR_LEN = 14\n",
        "K_ATR   = 2.0    # a \"win\" = price moves K_ATR x ATR in your direction first (same as the CNN test)\n",
        "HORIZON = 12     # ...within this many bars\n",
        "COST    = 0.05   # rough cost of one trade (spread) in the same units as the results table. Adjust to your broker."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "c_fFbmBP1CW7"
      },
      "execution_count": 2,
      "outputs": [],
      "source": [
        "# CELL 2 - FUNCTIONS: load data, indicators, labels, live prediction (nothing prints here)\n",
        "def load_symbol_csv(path):\n",
        "    df = pd.read_csv(path)\n",
        "    df.columns = [c.lower() for c in df.columns]\n",
        "    df[\"time\"] = pd.to_datetime(df[\"time\"])\n",
        "    df = df.sort_values(\"time\").reset_index(drop=True)\n",
        "    return df[[\"time\", \"open\", \"high\", \"low\", \"close\"]].dropna().reset_index(drop=True)\n",
        "\n",
        "def atr(df, n=ATR_LEN):\n",
        "    pc = df[\"close\"].shift(1)\n",
        "    tr = pd.concat([df[\"high\"] - df[\"low\"], (df[\"high\"] - pc).abs(), (df[\"low\"] - pc).abs()], axis=1).max(axis=1)\n",
        "    return tr.rolling(n).mean()\n",
        "\n",
        "def make_features(df):\n",
        "    \"\"\"Indicators that only use the PAST. Most are measured in ATR units, so EURUSD and USDJPY look comparable.\n",
        "    Used for BOTH training and live prediction.\"\"\"\n",
        "    c, a = df[\"close\"], atr(df)\n",
        "    f = pd.DataFrame(index=df.index)\n",
        "    for n in (1, 3, 6, 12, 24):                                   # momentum over different lengths\n",
        "        f[f\"ret_{n}\"] = (c - c.shift(n)) / a\n",
        "    for n in (20, 50):                                            # distance from moving averages\n",
        "        f[f\"dist_sma{n}\"] = (c - c.rolling(n).mean()) / a\n",
        "    delta = c.diff()\n",
        "    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()\n",
        "    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()\n",
        "    f[\"rsi\"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))\n",
        "    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()\n",
        "    f[\"macd_hist\"] = (macd - macd.ewm(span=9, adjust=False).mean()) / a\n",
        "    f[\"atr_pct\"] = a / c                                          # how volatile the market is\n",
        "    f[\"vol_regime\"] = a / a.rolling(100).mean()                   # volatile vs its own normal\n",
        "    rng = (df[\"high\"] - df[\"low\"]).replace(0, np.nan)             # shape of the last candle\n",
        "    f[\"body\"] = (df[\"close\"] - df[\"open\"]) / rng\n",
        "    f[\"upper_wick\"] = (df[\"high\"] - df[[\"open\", \"close\"]].max(axis=1)) / rng\n",
        "    f[\"lower_wick\"] = (df[[\"open\", \"close\"]].min(axis=1) - df[\"low\"]) / rng\n",
        "    hi, lo = df[\"high\"].rolling(20).max(), df[\"low\"].rolling(20).min()\n",
        "    f[\"range_pos\"] = (c - lo) / (hi - lo).replace(0, np.nan)      # where price sits in its 20-bar range\n",
        "    hour = df[\"time\"].dt.hour                                     # trading session (London / New York...)\n",
        "    f[\"hour_sin\"], f[\"hour_cos\"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)\n",
        "    f[\"dow\"] = df[\"time\"].dt.dayofweek\n",
        "    return f\n",
        "\n",
        "def label_at(high, low, close, i, a):\n",
        "    \"\"\"0 = buy (price rose K*ATR first), 2 = sell (fell K*ATR first), 1 = hold (neither within HORIZON bars).\"\"\"\n",
        "    up, dn = close[i] + K_ATR * a, close[i] - K_ATR * a\n",
        "    for j in range(i + 1, min(i + 1 + HORIZON, len(close))):\n",
        "        hit_up, hit_dn = high[j] >= up, low[j] <= dn\n",
        "        if hit_up and hit_dn:\n",
        "            return 1\n",
        "        if hit_up:\n",
        "            return 0\n",
        "        if hit_dn:\n",
        "            return 2\n",
        "    return 1\n",
        "\n",
        "def build_symbol(df):\n",
        "    f = make_features(df)\n",
        "    a = atr(df).to_numpy()\n",
        "    h, l, c = df[\"high\"].to_numpy(), df[\"low\"].to_numpy(), df[\"close\"].to_numpy()\n",
        "    y = np.full(len(df), -1)\n",
        "    for i in range(len(df) - HORIZON - 1):\n",
        "        if np.isfinite(a[i]) and a[i] > 0:\n",
        "            y[i] = label_at(h, l, c, i, a[i])\n",
        "    f[\"y\"] = y\n",
        "    f = f[f[\"y\"] >= 0].replace([np.inf, -np.inf], np.nan).dropna()\n",
        "    return f\n",
        "\n",
        "def split_by_time(n, train=0.70, val=0.15):\n",
        "    \"\"\"Oldest -> train, middle -> validation, newest -> test (with a gap so nothing leaks).\"\"\"\n",
        "    gap = HORIZON + 1\n",
        "    a, b = int(n * train), int(n * (train + val))\n",
        "    return slice(0, a - gap), slice(a, b - gap), slice(b, n)\n",
        "\n",
        "FEATURE_COLS = [c for c in make_features(\n",
        "    pd.DataFrame({\"time\": pd.date_range(\"2024-01-01\", periods=200, freq=\"h\"),\n",
        "                  \"open\": 1.0, \"high\": 1.0, \"low\": 1.0, \"close\": 1.0})).columns]\n",
        "\n",
        "def predict_signal(df_recent, bundle):\n",
        "    \"\"\"df_recent: the last ~250 CLOSED candles (time, open, high, low, close), newest last. Returns (SIGNAL, confidence, probs).\"\"\"\n",
        "    d = df_recent.rename(columns=str.lower).copy()\n",
        "    d[\"time\"] = pd.to_datetime(d[\"time\"])\n",
        "    row = make_features(d).iloc[[-1]][bundle[\"features\"]]\n",
        "    p = bundle[\"model\"].predict_proba(row)[0]\n",
        "    k = int(p.argmax())\n",
        "    return bundle[\"class_names\"][k].upper(), float(p[k]), p"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "BUILDING AND TEST DATA DEPLOYMET"
      ],
      "metadata": {
        "id": "cJCcuE7x7bfH"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E_RItQQx1CXA",
        "outputId": "8aa424a3-d81b-4969-e124-49213a531b84"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "AUDUSD_H1.csv: 49875 rows\n",
            "EURUSD_H1.csv: 49875 rows\n",
            "GBPUSD_H1.csv: 49875 rows\n",
            "USDCAD_H1.csv: 49875 rows\n",
            "USDJPY_H1.csv: 49875 rows\n",
            "train n=174495  buy=57376  hold=59267  sell=57852\n",
            "val   n= 37340  buy=12450  hold=12561  sell=12329\n",
            "test  n= 37410  buy=12004  hold=12777  sell=12629\n",
            "features used: 18\n"
          ]
        }
      ],
      "source": [
        "# CELL 3 - BUILD THE DATASET (fast: no pictures, usually under a minute)\n",
        "files = sorted(glob.glob(os.path.join(DATA_DIR, \"*.csv\")))\n",
        "assert files, f\"No CSV files found in {DATA_DIR}\"\n",
        "\n",
        "parts = {\"tr\": [], \"va\": [], \"te\": []}\n",
        "for fpath in files:\n",
        "    data = build_symbol(load_symbol_csv(fpath))\n",
        "    s_tr, s_va, s_te = split_by_time(len(data))\n",
        "    parts[\"tr\"].append(data.iloc[s_tr]); parts[\"va\"].append(data.iloc[s_va]); parts[\"te\"].append(data.iloc[s_te])\n",
        "    print(f\"{os.path.basename(fpath)}: {len(data)} rows\")\n",
        "\n",
        "train, val, test = [pd.concat(parts[k]) for k in (\"tr\", \"va\", \"te\")]\n",
        "Xtr, ytr = train[FEATURE_COLS], train[\"y\"].to_numpy()\n",
        "Xva, yva = val[FEATURE_COLS],   val[\"y\"].to_numpy()\n",
        "Xte, yte = test[FEATURE_COLS],  test[\"y\"].to_numpy()\n",
        "\n",
        "for name, yy in [(\"train\", ytr), (\"val\", yva), (\"test\", yte)]:\n",
        "    c = np.bincount(yy, minlength=3)\n",
        "    print(f\"{name:5s} n={len(yy):6d}  buy={c[0]}  hold={c[1]}  sell={c[2]}\")\n",
        "print(\"features used:\", len(FEATURE_COLS))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "MODEL TRAINING (CONVOLUTIONAL NEURAL NETWORK)"
      ],
      "metadata": {
        "id": "K9v1PZOq7Twi"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DYcOGLIl1CXF",
        "outputId": "2957ba84-ea9b-4369-ce71-cf70c94280f1"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "max_iter=  50   validation log-loss = 1.0083   (lower is better; guessing = 1.0986)\n",
            "max_iter= 100   validation log-loss = 1.0088   (lower is better; guessing = 1.0986)\n",
            "max_iter= 200   validation log-loss = 1.0097   (lower is better; guessing = 1.0986)\n",
            "max_iter= 400   validation log-loss = 1.0118   (lower is better; guessing = 1.0986)\n",
            "kept the model with log-loss 1.0083\n"
          ]
        }
      ],
      "source": [
        "# CELL 4 - TRAIN (tries a few sizes and keeps the one that does best on the validation period)\n",
        "from sklearn.ensemble import HistGradientBoostingClassifier\n",
        "from sklearn.metrics import log_loss\n",
        "\n",
        "best_model, best_ll = None, 1e9\n",
        "for n_iter in (50, 100, 200, 400):\n",
        "    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=n_iter,\n",
        "                                       min_samples_leaf=200, l2_regularization=1.0, random_state=1)\n",
        "    m.fit(Xtr, ytr)\n",
        "    ll = log_loss(yva, m.predict_proba(Xva), labels=[0, 1, 2])\n",
        "    print(f\"max_iter={n_iter:4d}   validation log-loss = {ll:.4f}   (lower is better; guessing = 1.0986)\")\n",
        "    if ll < best_ll:\n",
        "        best_model, best_ll = m, ll\n",
        "model = best_model\n",
        "print(\"kept the model with log-loss\", round(best_ll, 4))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "VERDICTING (READING ON UNSEEN DATA)"
      ],
      "metadata": {
        "id": "B4EgFsUF7Ijb"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ma0J-Q1p1CXH",
        "outputId": "f800330e-1e32-4726-c536-14aa58d8f46a"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "test accuracy: 0.473   | always-majority-class baseline: 0.342\n",
            "confusion matrix (rows = truth, cols = predicted; order buy/hold/sell):\n",
            "[[3912 3541 4551]\n",
            " [1834 8775 2168]\n",
            " [3809 3809 5011]]\n",
            "\n",
            "blind buy-every-bar: -0.017    blind sell-every-bar: +0.017\n",
            "expect = average result per signal (+1 win, -1 loss, 0 timeout).  luck+- = how far luck alone could push it.\n",
            " thr  side      n  right%  wrong%   expect   blind  luck+-  after cost\n",
            "0.40   buy   6565   43.2%   42.8%   +0.004  -0.017   0.079      -0.046\n",
            "0.40  sell   8907   44.4%   40.6%   +0.038  +0.017   0.068      -0.012\n",
            "0.45   buy   3103   45.6%   45.6%   -0.000  -0.017   0.119      -0.050\n",
            "0.45  sell   4669   46.6%   42.7%   +0.039  +0.017   0.096      -0.011\n",
            "0.50   buy    122   45.9%   46.7%   -0.008  -0.017   0.604      -0.058\n",
            "0.50  sell    268   51.1%   38.4%   +0.127  +0.017   0.397      +0.077\n",
            "0.55   buy      1  100.0%    0.0%   +1.000  -0.017   0.000      +0.950\n",
            "0.55  sell     31   41.9%   38.7%   +0.032  +0.017   1.117      -0.018\n",
            "0.60   buy      0\n",
            "0.60  sell      5    0.0%  100.0%   -1.000  +0.017   0.000      -1.050\n",
            "\n",
            "VERDICT: no reliable edge found (nothing beats the blind baseline + luck + costs with 500+ signals)\n",
            "\n",
            "Which indicators mattered most (bigger = more useful):\n",
            "  vol_regime   +0.05518\n",
            "  hour_sin     +0.04465\n",
            "  hour_cos     +0.03113\n",
            "  range_pos    +0.01777\n",
            "  atr_pct      +0.00362\n",
            "  dow          +0.00357\n",
            "  dist_sma20   +0.00248\n",
            "  rsi          +0.00176\n"
          ]
        }
      ],
      "source": [
        "# CELL 5 - HONEST TEST on the newest data the model has never seen\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.inspection import permutation_importance\n",
        "\n",
        "probs = model.predict_proba(Xte)\n",
        "pred = probs.argmax(1)\n",
        "print(f\"test accuracy: {(pred == yte).mean():.3f}   | always-majority-class baseline: {np.bincount(yte).max() / len(yte):.3f}\")\n",
        "print(\"confusion matrix (rows = truth, cols = predicted; order buy/hold/sell):\")\n",
        "print(confusion_matrix(yte, pred, labels=[0, 1, 2]))\n",
        "\n",
        "# \"blind\" = what you would earn by taking that side on EVERY bar. A signal only shows skill if it beats this.\n",
        "blind = {0: ((yte == 0).sum() - (yte == 2).sum()) / len(yte),\n",
        "         2: ((yte == 2).sum() - (yte == 0).sum()) / len(yte)}\n",
        "print(f\"\\nblind buy-every-bar: {blind[0]:+.3f}    blind sell-every-bar: {blind[2]:+.3f}\")\n",
        "print(\"expect = average result per signal (+1 win, -1 loss, 0 timeout).  luck+- = how far luck alone could push it.\")\n",
        "print(f\"{'thr':>4} {'side':>5} {'n':>6} {'right%':>7} {'wrong%':>7} {'expect':>8} {'blind':>7} {'luck+-':>7} {'after cost':>11}\")\n",
        "real_edge = []\n",
        "for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:\n",
        "    for side, name, opp in [(0, \"buy\", 2), (2, \"sell\", 0)]:\n",
        "        m = (pred == side) & (probs[:, side] >= thr)\n",
        "        n = int(m.sum())\n",
        "        if n == 0:\n",
        "            print(f\"{thr:4.2f} {name:>5} {0:6d}\"); continue\n",
        "        outcome = np.where(yte[m] == side, 1.0, np.where(yte[m] == opp, -1.0, 0.0))\n",
        "        e = outcome.mean()\n",
        "        luck = 2 * outcome.std() / np.sqrt(max(n / HORIZON, 1))   # neighbouring bars share outcomes, so count n / HORIZON\n",
        "        print(f\"{thr:4.2f} {name:>5} {n:6d} {(yte[m] == side).mean() * 100:6.1f}% {(yte[m] == opp).mean() * 100:6.1f}% \"\n",
        "              f\"{e:+8.3f} {blind[side]:+7.3f} {luck:7.3f} {e - COST:+11.3f}\")\n",
        "        if n >= 500 and (e - blind[side]) > luck and (e - COST) > 0:\n",
        "            real_edge.append((thr, name, n, round(float(e - COST), 3)))\n",
        "\n",
        "print(\"\\nVERDICT:\", (\"possible edge at \" + str(real_edge) + \" -> NOT proof; test on a demo account next\")\n",
        "      if real_edge else \"no reliable edge found (nothing beats the blind baseline + luck + costs with 500+ signals)\")\n",
        "\n",
        "print(\"\\nWhich indicators mattered most (bigger = more useful):\")\n",
        "sub = np.random.default_rng(0).choice(len(Xva), size=min(5000, len(Xva)), replace=False)\n",
        "pi = permutation_importance(model, Xva.iloc[sub], yva[sub], n_repeats=3, random_state=0, scoring=\"neg_log_loss\")\n",
        "for k in np.argsort(pi.importances_mean)[::-1][:8]:\n",
        "    print(f\"  {FEATURE_COLS[k]:12s} {pi.importances_mean[k]:+.5f}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#CHECKING THE RESULT PER PAIR\n",
        "sizes = [len(x) for x in parts[\"te\"]]\n",
        "names = [os.path.basename(f).split(\"_\")[0] for f in files]\n",
        "print(f\"{'pair':8s} {'n':>6} {'acc':>6}\")\n",
        "start = 0\n",
        "for name, size in zip(names, sizes):\n",
        "    p, y = probs[start:start + size], yte[start:start + size]\n",
        "    start += size\n",
        "    pr = p.argmax(1)\n",
        "    line = f\"{name:8s} {size:6d} {(pr == y).mean():6.3f}\"\n",
        "    for side, opp, label in [(0, 2, \"buy\"), (2, 0, \"sell\")]:\n",
        "        m = (pr == side) & (p[:, side] >= 0.45)\n",
        "        e = np.where(y[m] == side, 1.0, np.where(y[m] == opp, -1.0, 0.0)).mean() if m.sum() else float(\"nan\")\n",
        "        b = ((y == side).sum() - (y == opp).sum()) / len(y)\n",
        "        line += f\"   {label}: n={int(m.sum()):5d} expect={e:+.3f} blind={b:+.3f}\"\n",
        "    print(line)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8gTwLuSO4_QL",
        "outputId": "d584c586-c3b9-43bb-e7b4-edd7f8060b34"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "pair          n    acc\n",
            "AUDUSD     7482  0.427   buy: n=  477 expect=+0.006 blind=-0.032   sell: n=  613 expect=+0.044 blind=+0.032\n",
            "EURUSD     7482  0.503   buy: n=  723 expect=-0.047 blind=-0.021   sell: n= 1002 expect=+0.073 blind=+0.021\n",
            "GBPUSD     7482  0.518   buy: n=  793 expect=-0.050 blind=-0.020   sell: n=  965 expect=+0.089 blind=+0.020\n",
            "USDCAD     7482  0.517   buy: n=  512 expect=+0.053 blind=+0.018   sell: n= 1185 expect=-0.013 blind=-0.018\n",
            "USDJPY     7482  0.400   buy: n=  598 expect=+0.072 blind=-0.029   sell: n=  904 expect=+0.011 blind=+0.029\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "MODEL SAVING\n"
      ],
      "metadata": {
        "id": "vfkUlogQ6-Na"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SHujTcKD1CXJ",
        "outputId": "ce03e066-8fc5-4ee6-cabe-7b3bd698a456"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "saved: /content/drive/MyDrive/indicator_model.joblib\n"
          ]
        }
      ],
      "source": [
        "# CELL 6 - SAVE THE MODEL (after this you never need to retrain just to run the bot cells)\n",
        "joblib.dump({\"model\": model, \"features\": FEATURE_COLS, \"class_names\": CLASS_NAMES,\n",
        "             \"cfg\": {\"ATR_LEN\": ATR_LEN, \"K_ATR\": K_ATR, \"HORIZON\": HORIZON}}, MODEL_PATH)\n",
        "print(\"saved:\", MODEL_PATH)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Bot Setting and llm integration\n"
      ],
      "metadata": {
        "id": "sWsx5tZR6233"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l5G087Of1CXL",
        "outputId": "57167cc0-b95c-4e26-aa7d-8385ae8e0b6d"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Groq API key: ··········\n",
            "Gemini API key: ··········\n",
            "Telegram bot token: ··········\n",
            "Telegram chat id: ··········\n",
            "model loaded; features: 18\n"
          ]
        }
      ],
      "source": [
        "# CELL 7 - BOT SETTINGS, KEYS, LOAD THE SAVED MODEL\n",
        "# If the session restarted: run Cells 1 and 2 first, then this cell (no retraining needed).\n",
        "!pip install -q groq google-genai\n",
        "\n",
        "missing = [n for n in ('joblib', 'MODEL_PATH', 'load_symbol_csv', 'make_features', 'predict_signal') if n not in globals()]\n",
        "assert not missing, f'Run Cells 1 and 2 first. Missing: {missing}'\n",
        "\n",
        "import re, time, requests\n",
        "from getpass import getpass\n",
        "from groq import Groq\n",
        "from google import genai\n",
        "from google.genai import types\n",
        "\n",
        "# ---- the only numbers you should need to change ----\n",
        "MIN_CONFIDENCE = 0.45          # model must be at least this sure\n",
        "MIN_MARGIN     = 0.10          # top choice must beat the runner-up by at least this much\n",
        "USE_LLM_VETO   = True          # asking  Groq / Gemini to approve the trade\n",
        "SEND_TELEGRAM  = True\n",
        "GROQ_MODEL     = \"openai/gpt-oss-120b\"\n",
        "GEMINI_MODEL   = \"gemini-3.5-flash-lite\"\n",
        "\n",
        "# ---- keys (asked once; skipped if the name already exists in this session) ----\n",
        "if \"GROQ_API_KEY\" not in globals():    GROQ_API_KEY    = getpass(\"Groq API key: \").strip()\n",
        "if \"GEMINI_API_KEY\" not in globals():  GEMINI_API_KEY  = getpass(\"Gemini API key: \").strip()\n",
        "if \"TELEGRAM_TOKEN\" not in globals():  TELEGRAM_TOKEN  = getpass(\"Telegram bot token: \").strip()\n",
        "if \"CHAT_ID\" not in globals():         CHAT_ID         = getpass(\"Telegram chat id: \").strip()\n",
        "\n",
        "groq_client   = Groq(api_key=GROQ_API_KEY)          # Make sure that nothing will crush with a new client\n",
        "gemini_client = genai.Client(api_key=GEMINI_API_KEY)\n",
        "\n",
        "bundle = joblib.load(MODEL_PATH)                    #Loading and using the model saved\n",
        "print(\"model loaded; features:\", len(bundle[\"features\"]))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "INTEGRATING THE LLM APIs (GROQ AND GEMINI)"
      ],
      "metadata": {
        "id": "Oht9lgC88vzW"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "v0CZwSVg1CXN"
      },
      "execution_count": 10,
      "outputs": [],
      "source": [
        "# CELL 8 - LLM CHECK (Groq first, Gemini as backup). Any problem = veto, never a trade.\n",
        "LLM_SYSTEM = (\n",
        "    \"You are a cautious forex risk reviewer. A statistical model proposes a trade. \"\n",
        "    \"Reply with ONLY a JSON object: \"\n",
        "    '{\"agree\": true or false, \"risk\": \"LOW\" or \"MEDIUM\" or \"HIGH\", \"reason\": \"max 25 words\"}. '\n",
        "    \"Set agree=false if the recent candles or indicators clearly contradict the trade, or risk is HIGH. \"\n",
        "    \"Use only the data given. Do not invent news or prices.\"\n",
        ")\n",
        "\n",
        "def make_llm_summary(df_recent, signal, conf):\n",
        "    d = df_recent.rename(columns=str.lower)\n",
        "    f = make_features(d)[[\"ret_3\", \"ret_12\", \"dist_sma20\", \"dist_sma50\", \"rsi\", \"macd_hist\", \"vol_regime\"]]\n",
        "    return json.dumps({\n",
        "        \"proposed_trade\": signal,\n",
        "        \"model_confidence\": round(conf, 3),\n",
        "        \"last_10_candles_ohlc\": d[[\"open\", \"high\", \"low\", \"close\"]].tail(10).round(5).values.tolist(),\n",
        "        \"last_5_indicator_rows\": f.tail(5).round(3).to_dict(\"records\"),\n",
        "        \"note\": \"indicators are in ATR units; vol_regime above 1 means more volatile than normal\",\n",
        "    })\n",
        "\n",
        "def parse_llm_json(text):\n",
        "    m = re.search(r\"\\{.*\\}\", text or \"\", re.S)\n",
        "    if not m:\n",
        "        return None\n",
        "    try:\n",
        "        out = json.loads(m.group(0))\n",
        "    except Exception:\n",
        "        return None\n",
        "    return out if isinstance(out.get(\"agree\"), bool) else None\n",
        "\n",
        "def ask_groq(summary):\n",
        "    r = groq_client.chat.completions.create(\n",
        "        model=GROQ_MODEL, temperature=0, max_tokens=1000,\n",
        "        response_format={\"type\": \"json_object\"},\n",
        "        messages=[{\"role\": \"system\", \"content\": LLM_SYSTEM}, {\"role\": \"user\", \"content\": summary}])\n",
        "    return parse_llm_json(r.choices[0].message.content)\n",
        "\n",
        "def ask_gemini(summary):\n",
        "    r = gemini_client.models.generate_content(\n",
        "        model=GEMINI_MODEL, contents=summary,\n",
        "        config=types.GenerateContentConfig(system_instruction=LLM_SYSTEM, temperature=0,\n",
        "                                           response_mime_type=\"application/json\", max_output_tokens=1000))\n",
        "    return parse_llm_json(r.text)\n",
        "\n",
        "def llm_check(df_recent, signal, conf):\n",
        "    \"\"\"Returns (ok, notes). ok=False means VETO.\"\"\"\n",
        "    summary = make_llm_summary(df_recent, signal, conf)\n",
        "    for name, fn in ((\"groq\", ask_groq), (\"gemini\", ask_gemini)):\n",
        "        try:\n",
        "            out = fn(summary)\n",
        "            if out is not None:\n",
        "                ok = out[\"agree\"] and str(out.get(\"risk\", \"\")).upper() != \"HIGH\"\n",
        "                return ok, f\"{name}: agree={out['agree']} risk={out.get('risk')} - {out.get('reason', '')}\"\n",
        "        except Exception as e:\n",
        "            print(f\"{name} failed: {type(e).__name__}\")\n",
        "    return False, \"no valid LLM answer -> veto\""
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "DECISIONING AND TELEGRAM INTEGRATION\n"
      ],
      "metadata": {
        "id": "jakQb_J98h8k"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hOANwfDd1CXP"
      },
      "execution_count": 11,
      "outputs": [],
      "source": [
        "# CELL 9 - DECISION RULES + TELEGRAM\n",
        "def decide(df_recent, bundle):\n",
        "    \"\"\"df_recent: last ~250 CLOSED candles (time, open, high, low, close). Returns ('BUY'/'SELL'/'HOLD', info).\"\"\"\n",
        "    signal, conf, probs = predict_signal(df_recent, bundle)\n",
        "    info = {\"signal\": signal, \"confidence\": round(conf, 3),\n",
        "            \"probs\": [round(float(p), 3) for p in probs], \"note\": \"\"}\n",
        "    if signal == \"HOLD\":\n",
        "        info[\"note\"] = \"model says HOLD\"\n",
        "        return \"HOLD\", info\n",
        "    if conf < MIN_CONFIDENCE:\n",
        "        info[\"note\"] = f\"confidence {conf:.2f} below {MIN_CONFIDENCE}\"\n",
        "        return \"HOLD\", info\n",
        "    second = sorted(probs)[-2]                       # the runner-up class\n",
        "    if conf - second < MIN_MARGIN:\n",
        "        info[\"note\"] = f\"too close to call ({conf:.2f} vs {second:.2f}, need a gap of {MIN_MARGIN})\"\n",
        "        return \"HOLD\", info\n",
        "    if USE_LLM_VETO:\n",
        "        ok, notes = llm_check(df_recent, signal, conf)\n",
        "        info[\"note\"] = notes\n",
        "        if not ok:\n",
        "            return \"HOLD\", info\n",
        "    else:\n",
        "        info[\"note\"] = \"LLM check skipped\"\n",
        "    return signal, info\n",
        "\n",
        "def send_telegram(text):\n",
        "    r = requests.post(f\"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage\",\n",
        "                      data={\"chat_id\": CHAT_ID, \"text\": text}, timeout=30)\n",
        "    print(\"Telegram:\", \"sent\" if r.ok else f\"failed (status {r.status_code})\")\n",
        "    return r.ok"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "IYg9sqHt8Zvm"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "39FJt8Is1CXS",
        "outputId": "b49ed646-a8c1-4b38-9eb5-5ab05f622e05"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "AUDUSD: model would trade on 1.2% of the last 2000 bars (buy=4, sell=21)\n",
            "   (should be a small number, roughly under 10%. If it is large, raise MIN_CONFIDENCE or MIN_MARGIN in Cell 7)\n",
            "AUDUSD -> HOLD {'signal': 'SELL', 'confidence': 0.363, 'probs': [0.282, 0.355, 0.363], 'note': 'confidence 0.36 below 0.45'}\n"
          ]
        }
      ],
      "source": [
        "#  DRY RUN on the newest candles of one CSV file (no orders exist here; nothing is traded)\n",
        "csv_files = sorted(glob.glob(os.path.join(DATA_DIR, \"*.csv\")))\n",
        "df_all = load_symbol_csv(csv_files[0])\n",
        "symbol = os.path.basename(csv_files[0]).split(\"_\")[0]\n",
        "\n",
        "# 1) How often would the model even want to trade? (model only, no LLM, last 2000 bars)\n",
        "F = make_features(df_all).dropna().tail(2000)\n",
        "P = bundle[\"model\"].predict_proba(F[bundle[\"features\"]])\n",
        "top2 = np.sort(P, axis=1)[:, -2:]\n",
        "wants = (top2[:, 1] >= MIN_CONFIDENCE) & ((top2[:, 1] - top2[:, 0]) >= MIN_MARGIN) & (P.argmax(axis=1) != 1)\n",
        "print(f\"{symbol}: model would trade on {wants.mean() * 100:.1f}% of the last {len(F)} bars \"\n",
        "      f\"(buy={int((wants & (P.argmax(1) == 0)).sum())}, sell={int((wants & (P.argmax(1) == 2)).sum())})\")\n",
        "print(\"   (should be a small number, roughly under 10%. If it is large, raise MIN_CONFIDENCE or MIN_MARGIN in Cell 7)\")\n",
        "\n",
        "# 2) The full decision on the newest candles\n",
        "decision, info = decide(df_all.tail(250), bundle)\n",
        "print(symbol, \"->\", decision, info)\n",
        "if SEND_TELEGRAM and decision != \"HOLD\":\n",
        "    send_telegram(f\"{symbol} {decision} (model {info['confidence'] * 100:.0f}%)\\n{info['note']}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TESTING THE LLM AND SENDING THE SIGNAL TO TELEGRAM"
      ],
      "metadata": {
        "id": "qnlrlsbt8P0X"
      }
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I-CnEItZ1CXT",
        "outputId": "61a5b2d0-b6eb-4531-9ba4-f97cfa0113c3"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "bars where the model wanted a clear trade (0.40+ confidence): 47\n",
            "2026-11-11 11:00:00 -> SELL {'signal': 'SELL', 'confidence': 0.533, 'probs': [0.369, 0.097, 0.533], 'note': 'groq: agree=True risk=MEDIUM - Recent candles show slight downtrend; RSI falling below 50, price below SMA20, but MACD_hist still positive—moderate risk.'}\n",
            "Telegram: sent\n"
          ]
        }
      ],
      "source": [
        "#  TESTINING  THE LLM + TELEGRAM PATH on a past bar where the model wanted to trade\n",
        "F = make_features(df_all).dropna().tail(2000)\n",
        "P = bundle[\"model\"].predict_proba(F[bundle[\"features\"]])\n",
        "top2 = np.sort(P, axis=1)[:, -2:]\n",
        "rows = np.where((P.argmax(axis=1) != 1) & (top2[:, 1] >= 0.40) & ((top2[:, 1] - top2[:, 0]) >= MIN_MARGIN))[0]\n",
        "print(\"bars where the model wanted a clear trade (0.40+ confidence):\", len(rows))\n",
        "\n",
        "if len(rows) > 0:\n",
        "    pos = int(F.index[rows[-1]])                  # the most recent such bar\n",
        "    past = df_all.iloc[pos - 249: pos + 1]        # the 250 candles up to that bar\n",
        "    saved = MIN_CONFIDENCE\n",
        "    MIN_CONFIDENCE = 0.40                         # only for this test\n",
        "    decision, info = decide(past, bundle)\n",
        "    MIN_CONFIDENCE = saved\n",
        "    print(df_all[\"time\"].iloc[pos], \"->\", decision, info)\n",
        "    if SEND_TELEGRAM:\n",
        "        send_telegram(f\"TEST (past bar {df_all['time'].iloc[pos]}) {decision}\\n{info['note']}\")\n",
        "else:\n",
        "    print(\"no such bar in the last 2000; this is fine, it just means the model is very cautious\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install -q trafilatura feedparser\n"
      ],
      "metadata": {
        "id": "qXCMPbj3uVlT"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ============================================================\n",
        "# CELL 12 — NEWS CONFIG + LLM CLIENT (Groq / Llama 3.1 8B)\n",
        "# ============================================================\n",
        "\n",
        "import feedparser\n",
        "import trafilatura\n",
        "from datetime import datetime, timezone, timedelta\n",
        "\n",
        "# ---------- RSS SOURCES ----------\n",
        "RSS_FEEDS = [\n",
        "\n",
        "    (\"forexlive\",  \"https://www.forexlive.com/feed/news\"),\n",
        "\n",
        "\n",
        "]\n",
        "\n",
        "# ---------- NEWS CONSTANTS ----------\n",
        "NEWS_WINDOWS = {\n",
        "    \"sentiment_1h\":  timedelta(hours=1),\n",
        "    \"sentiment_4h\":  timedelta(hours=4),\n",
        "    \"sentiment_24h\": timedelta(hours=24),\n",
        "}\n",
        "NEWS_IMPACT_MAP = {\"LOW\": 0.3, \"MEDIUM\": 0.6, \"HIGH\": 1.0}\n",
        "NEWS_MAX_ARTICLE_CHARS = 4000\n",
        "NEWS_MERGE_TOLERANCE = timedelta(hours=4)\n",
        "\n",
        "# ---------- LLM CLIENT ----------\n",
        "NEWS_LLM_MODEL = \"openai/gpt-oss-20b\"\n",
        "\n",
        "if \"groq_client\" in globals():\n",
        "    news_llm = groq_client\n",
        "else:\n",
        "    from groq import Groq\n",
        "    news_llm = Groq(api_key=GROQ_API_KEY)\n",
        "\n",
        "# ---------- PREFLIGHT ----------\n",
        "missing = [n for n in (\"CLASS_NAMES\",) if n not in globals()]\n",
        "assert not missing, f\"Run Cell 1 first. Missing: {missing}\"\n",
        "\n",
        "print(\"✅ Cell 12 ready\")\n",
        "print(f\"   RSS feeds: {len(RSS_FEEDS)} (full)\")\n",
        "print(f\"   LLM model: {NEWS_LLM_MODEL} (Groq free tier)\")\n",
        "print(f\"   News windows: {list(NEWS_WINDOWS.keys())}\")\n",
        "print(f\"   Impact map keys: {list(NEWS_IMPACT_MAP.keys())}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZWqWby9U-plZ",
        "outputId": "707f6912-23e1-4c5d-8386-340697a7b9a9"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Cell 12 ready\n",
            "   RSS feeds: 1 (full)\n",
            "   LLM model: openai/gpt-oss-20b (Groq free tier)\n",
            "   News windows: ['sentiment_1h', 'sentiment_4h', 'sentiment_24h']\n",
            "   Impact map keys: ['LOW', 'MEDIUM', 'HIGH']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -q 'urllib3>=2.8.0'"
      ],
      "metadata": {
        "id": "yyV3vZ8FtbuB"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ============================================================\n",
        "# CELL 13 — RSS FETCH + TRAFILATURA EXTRACT\n",
        "# Zero LLM calls. Produces news_df.\n",
        "# Columns: published_at_utc, source, title, url, summary, body\n",
        "# Each feed isolated in try/except — one dead feed won't kill the cell.\n",
        "# ============================================================\n",
        "\n",
        "import time\n",
        "from datetime import datetime, timezone\n",
        "\n",
        "def fetch_rss_entries(feed_url, source_name):\n",
        "    \"\"\"Fetch one RSS feed, return list of dicts.\"\"\"\n",
        "    try:\n",
        "        feed = feedparser.parse(feed_url)\n",
        "        entries = []\n",
        "        for e in feed.entries:\n",
        "            pub = getattr(e, \"published_parsed\", None)\n",
        "            if pub is None:\n",
        "                continue\n",
        "            published_at = datetime(*pub[:6], tzinfo=timezone.utc)\n",
        "            entries.append({\n",
        "                \"published_at_utc\": published_at,\n",
        "                \"source\": source_name,\n",
        "                \"title\": getattr(e, \"title\", \"\").strip(),\n",
        "                \"url\": getattr(e, \"link\", \"\").strip(),\n",
        "                \"summary\": getattr(e, \"summary\", \"\").strip(),\n",
        "            })\n",
        "        return entries\n",
        "    except Exception as e:\n",
        "        print(f\"  [{source_name}] RSS fetch failed: {type(e).__name__}: {e}\")\n",
        "        return []\n",
        "\n",
        "def extract_body(url):\n",
        "    \"\"\"Fetch article URL, extract main text via trafilatura. Returns '' on failure.\"\"\"\n",
        "    try:\n",
        "        downloaded = trafilatura.fetch_url(url)\n",
        "        if not downloaded:\n",
        "            return \"\"\n",
        "        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)\n",
        "        return (text or \"\")[:NEWS_MAX_ARTICLE_CHARS]\n",
        "    except Exception:\n",
        "        return \"\"\n",
        "\n",
        "# ---------- MAIN ----------\n",
        "rows = []\n",
        "for name, url in RSS_FEEDS:\n",
        "    print(f\"Fetching {name} ...\")\n",
        "    entries = fetch_rss_entries(url, name)\n",
        "    print(f\"  {len(entries)} entries\")\n",
        "    for e in entries:\n",
        "        e[\"body\"] = extract_body(e[\"url\"])\n",
        "        rows.append(e)\n",
        "    time.sleep(0.5)\n",
        "\n",
        "news_df = pd.DataFrame(rows)\n",
        "if len(news_df) > 0:\n",
        "    news_df = news_df.drop_duplicates(subset=[\"url\"]).reset_index(drop=True)\n",
        "    news_df = news_df[news_df[\"body\"].str.len() > 100].reset_index(drop=True)\n",
        "\n",
        "print(f\"\\n✅ Cell 13 done: {len(news_df)} articles (deduped, body > 100 chars)\")\n",
        "if len(news_df) > 0:\n",
        "    print(news_df[[\"published_at_utc\", \"source\", \"title\"]].head(5).to_string())\n",
        "else:\n",
        "    print(\"⚠️  No articles fetched. Check RSS_FEEDS in Cell 12 and errors above.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pBnH5M4ZsUFf",
        "outputId": "0de91f3e-a762-44d9-d02f-82da2cf35dfd"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fetching forexlive ...\n",
            "  25 entries\n",
            "\n",
            "✅ Cell 13 done: 25 articles (deduped, body > 100 chars)\n",
            "           published_at_utc     source                                                                                              title\n",
            "0 2026-11-20 07:40:50+00:00  forexlive                                                                What are the main events for today?\n",
            "1 2026-11-20 07:00:37+00:00  forexlive                                                UK October retail sales +0.7% vs +0.0% m/m expected\n",
            "2 2026-11-20 04:39:38+00:00  forexlive                     investingLive Asia-Pacific market news: Red Friday as Asia’s AI trade unravels\n",
            "3 2026-11-20 02:14:13+00:00  forexlive                    US envoys float NATO membership for Ukraine in exchange for Donbas, WSJ reports\n",
            "4 2026-11-20 01:16:31+00:00  forexlive  US approved nearly 900,000 Nvidia and AMD AI chips for China. Raises more questions than answers.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#GOOGLE NEWS RSS (LAST 6 MONTHS, WEEKLY WINDOWS)\n",
        "#Google News caps at ~100 items/query\n",
        "import time\n",
        "import urllib.parse\n",
        "from datetime import datetime, timezone, timedelta\n",
        "import feedparser\n",
        "import pandas as pd\n",
        "\n",
        "google_news_query = \"EURUSD OR GBPUSD OR USDJPY OR AUDUSD\"\n",
        "google_news_months = 6\n",
        "google_news_window_days = 7\n",
        "google_news_lang = \"en-US\"\n",
        "google_news_country = \"US\"\n",
        "\n",
        "def fetch_google_news_windows(query, after_date, before_date):\n",
        "    \"\"\" Fetch one date window. Return list of dicts. \"\"\"\n",
        "    q = f\"{query} after:{after_date} before:{before_date}\"\n",
        "    url = (\n",
        "        \"https://news.google.com/rss/search?\"\n",
        "        f\"q={urllib.parse.quote(q)}\"\n",
        "        f\"&hl={google_news_lang}&gl={google_news_country}&ceid={google_news_country}:en\"\n",
        "    )\n",
        "    try:\n",
        "        feed = feedparser.parse(url)\n",
        "        out = []\n",
        "        for e in feed.entries:\n",
        "            pub = getattr(e, \"published_parsed\", None)\n",
        "            if pub is None:\n",
        "                continue\n",
        "            out.append({\n",
        "                \"published_at_utc\": datetime(*pub[:6], tzinfo=timezone.utc),\n",
        "                \"source\": \"Google News\",\n",
        "                \"title\": getattr(e, \"title\", \"\").strip(),\n",
        "                \"url\": getattr(e, \"link\", \"\").strip(),\n",
        "                \"summary\": getattr(e, \"summary\", \"\").strip(),\n",
        "            })\n",
        "        return out, len(feed.entries)\n",
        "    except Exception as ex:\n",
        "        print(f\" window {after_date}..{before_date} failed: {type(ex).__name__}\")\n",
        "        return [], 0\n",
        "\n",
        "end_dt = datetime.now(timezone.utc)\n",
        "start_dt = end_dt - timedelta(days=google_news_months * 30)\n",
        "\n",
        "all_rows = []\n",
        "cur = start_dt\n",
        "window_count = 0\n",
        "\n",
        "while cur < end_dt:\n",
        "    win_end = min(cur + timedelta(days=google_news_window_days), end_dt)\n",
        "    rows, raw_count = fetch_google_news_windows(\n",
        "        google_news_query, cur.strftime(\"%Y-%m-%d\"), win_end.strftime(\"%Y-%m-%d\")\n",
        "    )\n",
        "    all_rows.extend(rows)\n",
        "    window_count += 1\n",
        "    flag = \" Truncated (100)\" if raw_count >= 100 else \"\"\n",
        "    print(f\" {cur.strftime('%Y-%m-%d')} → {win_end.strftime('%Y-%m-%d')}: {len(rows)} articles{flag}\")\n",
        "    cur = win_end\n",
        "    time.sleep(0.8)\n",
        "\n",
        "# data appending\n",
        "gn_df = pd.DataFrame(all_rows)\n",
        "if len(gn_df) > 0:\n",
        "    gn_df = gn_df.drop_duplicates(subset=[\"url\"]).reset_index(drop=True)\n",
        "    gn_df = gn_df[gn_df[\"title\"].str.len() > 10].reset_index(drop=True)\n",
        "    gn_df[\"body\"] = gn_df[\"summary\"]\n",
        "    gn_df = gn_df[[\"published_at_utc\", \"source\", \"title\", \"url\", \"summary\", \"body\"]]\n",
        "\n",
        "# Merging the two datasets\n",
        "if \"news_df\" in globals() and len(news_df) > 0:\n",
        "    news_df = pd.concat([news_df, gn_df], ignore_index=True)\n",
        "    news_df = news_df.drop_duplicates(subset=[\"url\"]).reset_index(drop=True)\n",
        "else:\n",
        "    news_df = gn_df\n",
        "\n",
        "print(f\"\\n✅ Fetching data done: +{len(gn_df)} from Google News, total news_df = {len(news_df)}\")\n",
        "if len(news_df) > 0:\n",
        "    print(f\"   date range: {news_df['published_at_utc'].min()} → {news_df['published_at_utc'].max()}\")\n",
        "    print(news_df[[\"published_at_utc\", \"title\"]].head(5).to_string())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r4liwTuxG14z",
        "outputId": "d8ccfa6d-a9a6-4602-dc9d-c4eb51b7b82a"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " 2026-05-24 → 2026-05-31: 6 articles\n",
            " 2026-05-31 → 2026-06-07: 8 articles\n",
            " 2026-06-07 → 2026-06-14: 10 articles\n",
            " 2026-06-14 → 2026-06-21: 4 articles\n",
            " 2026-06-21 → 2026-06-28: 7 articles\n",
            " 2026-06-28 → 2026-07-05: 11 articles\n",
            " 2026-07-05 → 2026-07-12: 7 articles\n",
            " 2026-07-12 → 2026-07-19: 9 articles\n",
            " 2026-07-19 → 2026-07-26: 6 articles\n",
            " 2026-07-26 → 2026-08-02: 8 articles\n",
            " 2026-08-02 → 2026-08-09: 10 articles\n",
            " 2026-08-09 → 2026-08-16: 7 articles\n",
            " 2026-08-16 → 2026-08-23: 10 articles\n",
            " 2026-08-23 → 2026-08-30: 4 articles\n",
            " 2026-08-30 → 2026-09-06: 8 articles\n",
            " 2026-09-06 → 2026-09-13: 4 articles\n",
            " 2026-09-13 → 2026-09-20: 2 articles\n",
            " 2026-09-20 → 2026-09-27: 24 articles\n",
            " 2026-09-27 → 2026-10-04: 19 articles\n",
            " 2026-10-04 → 2026-10-11: 34 articles\n",
            " 2026-10-11 → 2026-10-18: 17 articles\n",
            " 2026-10-18 → 2026-10-25: 10 articles\n",
            " 2026-10-25 → 2026-11-01: 15 articles\n",
            " 2026-11-01 → 2026-11-08: 16 articles\n",
            " 2026-11-08 → 2026-11-15: 25 articles\n",
            " 2026-11-15 → 2026-11-20: 100 articles Truncated (100)\n",
            "\n",
            "✅ Fetching data done: +371 from Google News, total news_df = 396\n",
            "   date range: 2026-05-24 07:00:00+00:00 → 2026-11-20 07:40:50+00:00\n",
            "           published_at_utc                                                                                              title\n",
            "0 2026-11-20 07:40:50+00:00                                                                What are the main events for today?\n",
            "1 2026-11-20 07:00:37+00:00                                                UK October retail sales +0.7% vs +0.0% m/m expected\n",
            "2 2026-11-20 04:39:38+00:00                     investingLive Asia-Pacific market news: Red Friday as Asia’s AI trade unravels\n",
            "3 2026-11-20 02:14:13+00:00                    US envoys float NATO membership for Ukraine in exchange for Donbas, WSJ reports\n",
            "4 2026-11-20 01:16:31+00:00  US approved nearly 900,000 Nvidia and AMD AI chips for China. Raises more questions than answers.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Refetching the the truncated windows\n",
        "# ============================================================\n",
        "# CELL 13c — RE-FETCH THE TWO TRUNCATED WINDOWS WITH 3-DAY SUBWINDOWS\n",
        "# ============================================================\n",
        "\n",
        "from datetime import date, timedelta\n",
        "\n",
        "truncated_windows = [\n",
        "    (date(2026, 11, 1),  date(2026, 11, 8)),\n",
        "    (date(2026, 11, 15), date(2026, 11, 20)),\n",
        "]\n",
        "\n",
        "extra_rows = []\n",
        "for start, end in truncated_windows:\n",
        "    cur = start\n",
        "    while cur < end:\n",
        "        sub_end = min(cur + timedelta(days=3), end)\n",
        "        rows, raw = fetch_google_news_windows(\n",
        "            google_news_query, cur.isoformat(), sub_end.isoformat()\n",
        "        )\n",
        "        extra_rows.extend(rows)\n",
        "        flag = \" ⚠️ STILL TRUNCATED\" if raw >= 100 else \"\"\n",
        "        print(f\"  {cur} → {sub_end}: {len(rows)} articles{flag}\")\n",
        "        cur = sub_end\n",
        "        time.sleep(0.8)\n",
        "\n",
        "extra_df = pd.DataFrame(extra_rows)\n",
        "if len(extra_df) > 0:\n",
        "    extra_df = extra_df.drop_duplicates(subset=[\"url\"]).reset_index(drop=True)\n",
        "    extra_df = extra_df[extra_df[\"title\"].str.len() > 10].reset_index(drop=True)\n",
        "    extra_df[\"body\"] = extra_df[\"summary\"]\n",
        "    extra_df = extra_df[[\"published_at_utc\", \"source\", \"title\", \"url\", \"summary\", \"body\"]]\n",
        "\n",
        "    news_df = pd.concat([news_df, extra_df], ignore_index=True)\n",
        "    news_df = news_df.drop_duplicates(subset=[\"url\"]).reset_index(drop=True)\n",
        "\n",
        "print(f\"\\n✅ Refetching done: +{len(extra_df) if len(extra_df) else 0}, total news_df = {len(news_df)}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TaJKGjbaRfDw",
        "outputId": "0cd690a0-2686-4aad-e475-97d9414f2f4c"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  2026-11-01 → 2026-11-04: 11 articles\n",
            "  2026-11-04 → 2026-11-07: 12 articles\n",
            "  2026-11-07 → 2026-11-08: 9 articles\n",
            "  2026-11-15 → 2026-11-18: 85 articles\n",
            "  2026-11-18 → 2026-11-20: 73 articles\n",
            "\n",
            "✅ Refetching done: +185, total news_df = 497\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#CHECKING DUPLICATIONS\n",
        "print(\"Total:\", len(news_df))\n",
        "print(\"Unique URLs:\", news_df[\"url\"].nunique())\n",
        "print(\"Unique titles (exact):\", news_df[\"title\"].nunique())\n",
        "\n",
        "# loose duplicate check: first 60 chars of title\n",
        "news_df[\"_tkey\"] = news_df[\"title\"].str.lower().str[:60]\n",
        "print(\"Unique title-prefixes:\", news_df[\"_tkey\"].nunique())\n",
        "del news_df[\"_tkey\"]\n",
        "\n",
        "\n",
        "print(\"Body length: min={}, median={}, max={}\".format(\n",
        "    news_df[\"body\"].str.len().min(),\n",
        "    int(news_df[\"body\"].str.len().median()),\n",
        "    news_df[\"body\"].str.len().max(),\n",
        "))\n",
        "gn_sample = news_df[news_df[\"source\"] == \"google_news\"].head(1)\n",
        "if len(gn_sample) > 0:\n",
        "    print(\"\\nSample Google News body:\\n\", gn_sample[\"body\"].iloc[0][:400])\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "39Ks-LEmSWbx",
        "outputId": "48c1a706-0f75-4a63-dde3-c13a6595c07c"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total: 497\n",
            "Unique URLs: 497\n",
            "Unique titles (exact): 475\n",
            "Unique title-prefixes: 456\n",
            "Body length: min=267, median=476, max=4000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Rows:\", len(news_df))\n",
        "print(\"Unique URLs:\", news_df[\"url\"].nunique())\n",
        "print(\"Body length: min={}, median={}, max={}\".format(\n",
        "    news_df[\"body\"].str.len().min(),\n",
        "    int(news_df[\"body\"].str.len().median()),\n",
        "    news_df[\"body\"].str.len().max(),\n",
        "))\n",
        "print(\"\\nSample body (first 400 chars of row 0):\\n\")\n",
        "print(news_df[\"body\"].iloc[0][:400])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AhPQqrVSvunD",
        "outputId": "c869a662-5463-4e11-93f3-b398251592ce"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Rows: 497\n",
            "Unique URLs: 497\n",
            "Body length: min=267, median=476, max=4000\n",
            "\n",
            "Sample body (first 400 chars of row 0):\n",
            "\n",
            "What are the main events for today?\n",
            "Summary\n",
            "- We don't have much on the agenda today as the focus will remain on growth concerns and US-Iran developments\n",
            "EUROPEAN SESSION\n",
            "In the European session, the main highlight was the UK Retail Sales report. The data showed a strong rebound in October, beating expectations across both headline and underlying measures, with volumes rising 0.7% m/m and sales ex\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ============================================================\n",
        "# NEWS SENTIMENT CLASSIFIER (Groq / gpt-oss-20b)\n",
        "# Each article -> {\"sentiment\": float, \"impact\": ..., \"currency\": ...}\n",
        "\n",
        "import json, re, os, time\n",
        "\n",
        "NEWS_CACHE_PATH = \"/content/drive/MyDrive/news_sentiment_cache.json\"\n",
        "\n",
        "NEWS_SYSTEM = (\n",
        "    \"You are a forex news sentiment classifier. \"\n",
        "    \"Return ONLY a JSON object with exactly these keys: \"\n",
        "    '{\"sentiment\": <float between -1.0 and 1.0>, '\n",
        "    '\"impact\": \"LOW\" or \"MEDIUM\" or \"HIGH\", '\n",
        "    '\"currency\": \"USD\" or \"EUR\" or \"GBP\" or \"JPY\" or \"AUD\" or \"CAD\" or \"CHF\" or \"NZD\"}. '\n",
        "    \"sentiment: negative = bearish for that currency, positive = bullish. \"\n",
        "    \"impact: likelihood this news moves the currency in the next 1-4 hours. \"\n",
        "    \"If the article is not about a specific currency (e.g. generic commodities, \"\n",
        "    \"equities, crypto), still pick the most plausible affected currency. \"\n",
        "    \"Use only the text given. Do not invent prices or events. \"\n",
        "    \"Output JSON only, no prose.\"\n",
        ")\n",
        "\n",
        "def parse_news_json(text):\n",
        "    \"\"\"Extract JSON, validate fields strictly. Returns dict or None.\"\"\"\n",
        "    if not text:\n",
        "        return None\n",
        "    m = re.search(r\"\\{.*\\}\", text, re.S)\n",
        "    if not m:\n",
        "        return None\n",
        "    try:\n",
        "        out = json.loads(m.group(0))\n",
        "    except Exception:\n",
        "        return None\n",
        "    s   = out.get(\"sentiment\")\n",
        "    imp = str(out.get(\"impact\", \"\")).strip().upper()\n",
        "    cur = str(out.get(\"currency\", \"\")).strip().upper()\n",
        "    if not isinstance(s, (int, float)) or not (-1.0 <= float(s) <= 1.0):\n",
        "        return None\n",
        "    if imp not in NEWS_IMPACT_MAP:\n",
        "        return None\n",
        "    if cur not in {\"USD\", \"EUR\", \"GBP\", \"JPY\", \"AUD\", \"CAD\", \"CHF\", \"NZD\"}:\n",
        "        return None\n",
        "    return {\"sentiment\": float(s), \"impact\": imp, \"currency\": cur}\n",
        "\n",
        "def classify_one(article):\n",
        "    \"\"\"Call Groq. Retries once on rate limit. Returns dict or None.\"\"\"\n",
        "    text = f\"Title: {article['title']}\\n\\nBody: {article['body']}\"\n",
        "    for attempt in range(2):\n",
        "        try:\n",
        "            r = news_llm.chat.completions.create(\n",
        "                model=NEWS_LLM_MODEL,\n",
        "                messages=[\n",
        "                    {\"role\": \"system\", \"content\": NEWS_SYSTEM},\n",
        "                    {\"role\": \"user\",   \"content\": text + \"\\n\\nRespond with JSON only\"},\n",
        "                ],\n",
        "                temperature=0,\n",
        "\n",
        "                max_tokens=500,\n",
        "            )\n",
        "            return parse_news_json(r.choices[0].message.content)\n",
        "        except Exception as e:\n",
        "            if \"RateLimit\" in type(e).__name__ and attempt == 0:\n",
        "                time.sleep(5)\n",
        "                continue\n",
        "            print(f\"  LLM failed on {article['url'][:60]}: {type(e).__name__}\")\n",
        "            return None\n",
        "\n",
        "# ---------- cache ----------\n",
        "cache = {}\n",
        "if os.path.exists(NEWS_CACHE_PATH):\n",
        "    try:\n",
        "        with open(NEWS_CACHE_PATH) as f:\n",
        "            cache = json.load(f)\n",
        "        print(f\"Loaded {len(cache)} cached classifications\")\n",
        "    except Exception:\n",
        "        cache = {}\n",
        "\n",
        "# ---------- main ----------\n",
        "results = []\n",
        "for i, art in enumerate(news_df.itertuples(index=False)):\n",
        "    key = art.url\n",
        "    if key in cache:\n",
        "        results.append(cache[key])\n",
        "        continue\n",
        "    out = classify_one({\"title\": art.title, \"url\": art.url, \"body\": art.body})\n",
        "    if out is not None:\n",
        "        out[\"url\"] = key\n",
        "        out[\"published_at_utc\"] = art.published_at_utc.isoformat()\n",
        "        results.append(out)\n",
        "        cache[key] = out\n",
        "    else:\n",
        "        results.append(None)\n",
        "    if (i + 1) % 20 == 0:\n",
        "        print(f\"  classified {i+1}/{len(news_df)}\")\n",
        "    time.sleep(2.5)   # <-- throttle: ~40 req/min, under Groq free-tier limit\n",
        "    if (i + 1) % 50 == 0:\n",
        "      with open(NEWS_CACHE_PATH, \"w\") as f:\n",
        "        json.dump(cache, f)\n",
        "      print(f\" cache saved at {i + 1} / {len(news_df)}\")\n",
        "with open(NEWS_CACHE_PATH, \"w\") as f:\n",
        "    json.dump(cache, f)\n",
        "\n",
        "news_scored = pd.DataFrame([r for r in results if r is not None])\n",
        "news_scored[\"published_at_utc\"] = pd.to_datetime(news_scored[\"published_at_utc\"], utc=True)\n",
        "\n",
        "print(f\"\\n✅ Cell 14 done: {len(news_scored)}/{len(news_df)} articles classified\")\n",
        "if len(news_scored) > 0:\n",
        "    print(news_scored[[\"published_at_utc\", \"sentiment\", \"impact\", \"currency\"]].head(5).to_string())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Tug8QNdMdQmQ",
        "outputId": "7a2f5344-55cb-4c62-9187-d8431ac6859b"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Loaded 178 cached classifications\n",
            "  classified 60/497\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMi1AFBVV95cUxQQVF1Wmo: RateLimitError\n",
            "  classified 80/497\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMirgFBVV95cUxObHlYdXQ: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMikgFBVV95cUxObzdFTC1: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMiogFBVV95cUxQSnNaZUN: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMivwFBVV95cUxQRXVEb2o: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMiwgFBVV95cUxNWlVMY3h: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMisgFBVV95cUxQQXBNVXN: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMieEFVX3lxTE51NjhfenZ: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMisAFBVV95cUxQY2FXbDV: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMiwwFBVV95cUxNbDdqak5: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMifEFVX3lxTE9hUnlVUGx: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMihAJBVV95cUxQcDhjZ0J: RateLimitError\n",
            "  classified 100/497\n",
            " cache saved at 100 / 497\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMiqAFBVV95cUxNLVM2cm5: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMijgFBVV95cUxNd2w0T2l: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMitwFBVV95cUxQeGcxdzd: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMitgFBVV95cUxPal9feVY: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMisgFBVV95cUxPOE5oSXN: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMiuAFBVV95cUxQRlMxbVJ: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMitAFBVV95cUxOZkllT2c: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMivgFBVV95cUxNYjNMUlB: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMipwFBVV95cUxQaVhpVUx: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMioAFBVV95cUxOUVMwVW1: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMioAFBVV95cUxQc2dJeTN: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMirgFBVV95cUxQOFFsVkJ: RateLimitError\n",
            "  classified 120/497\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMipAFBVV95cUxNVjhtVGZ: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMikgFBVV95cUxPcy02SE5: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMixgFBVV95cUxOcjh3VF8: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMiwAFBVV95cUxNdEpIdkx: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMikwFBVV95cUxPRnBXQjN: RateLimitError\n",
            "  LLM failed on https://news.google.com/rss/articles/CBMipgFBVV95cUxOVWthalV: RateLimitError\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "KeyboardInterrupt\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Checking the data and the candle range\n",
        "print(\"Candles range:\", df_all[\"time\"].min(), \"→\", df_all[\"time\"].max())\n",
        "print(\"News range:   \", news_scored[\"published_at_utc\"].min(), \"→\", news_scored[\"published_at_utc\"].max())"
      ],
      "metadata": {
        "id": "MfA-2rd6zdiD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Aligning news-scored to any candles DataFrame via merge _asof\n",
        "#Returning a DataFrame indexed the same way as candles , with 5 columns\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "def align_news_to_candles(candles, news=news_scored):\n",
        "    \"\"\"\n",
        "    candles: DataFrame with a 'time' column (datetime64[ns, UTC] preferred)\n",
        "    news : news_scored from cell 14\n",
        "    Return : DataFrame with 5 features columns, indexed like the candles.\n",
        "    \"\"\"\n",
        "    candles = candles.copy()\n",
        "    if \"time\" not in candles.columns:\n",
        "        raise ValueError(\"candles must have a time column\")\n",
        "    c_times = pd.to_datetime(candles[\"time\"], utc=True)\n",
        "    candles = candles.assign(_candle_time=c_times).reset_index(drop=True)\n",
        "    if len(news) == 0:\n",
        "        out = pd.DataFrame({\n",
        "            \"news_sentiment_1h\": 0.0,\n",
        "            \"news_sentiment_4h\": 0.0,\n",
        "            \"news_sentiment_24h\": 0.0,\n",
        "            \"news_counts_1h\": 0,\n",
        "            \"news_has_data\": 0\n",
        "        }, index=candles.index)\n",
        "        return out\n",
        "\n",
        "    nws = news.copy()\n",
        "    nws[\"published_at_utc\"] = pd.to_datetime(nws[\"published_at_utc\"], utc=True)\n",
        "    nws = nws.sort_values(\"published_at_utc\").reset_index(drop=True)\n",
        "\n",
        "    nws[\"impact_w\"] = nws[\"impact\"].map(NEWS_IMPACT_MAP).astype(float)\n",
        "    nws[\"w_sent\"] = nws[\"sentiment\"] * nws[\"impact_w\"]\n",
        "\n",
        "    # Aligning each candle with the most recent prior news\n",
        "    left = candles.sort_values(\"_candle_time\").reset_index()\n",
        "    right = nws[[\"published_at_utc\", \"w_sent\", \"impact_w\"]]\n",
        "\n",
        "    # Making the news to aggregate\n",
        "    sents_1h, sents_4h, sents_24h = [], [], []\n",
        "    counts_1h = []\n",
        "    news_times = nws[\"published_at_utc\"].values\n",
        "    news_sents = nws[\"w_sent\"].values\n",
        "    imp_arr = nws[\"impact_w\"].values\n",
        "\n",
        "    for t in left[\"_candle_time\"].values:\n",
        "        for win_h, bucket in [(1, \"1h\"), (4, \"4h\"), (24, \"24h\")]:\n",
        "            lo = t - np.timedelta64(win_h, \"h\")\n",
        "            mask = (news_times >= lo) & (news_times <= t)\n",
        "            n_in = mask.sum()\n",
        "            if n_in == 0:\n",
        "                if bucket == \"1h\": sents_1h.append(0.0)\n",
        "                if bucket == \"4h\": sents_4h.append(0.0)\n",
        "                if bucket == \"24h\": sents_24h.append(0.0)\n",
        "                if bucket == \"1h\": counts_1h.append(0)\n",
        "            else:\n",
        "                wsum = news_sents[mask].sum()\n",
        "                isum = imp_arr[mask].sum()\n",
        "                val = (wsum / isum) if isum > 0 else 0.0\n",
        "                if bucket == \"1h\": sents_1h.append(val)\n",
        "                if bucket == \"4h\": sents_4h.append(val)\n",
        "                if bucket == \"24h\": sents_24h.append(val)\n",
        "                if bucket == \"1h\": counts_1h.append(n_in)\n",
        "\n",
        "    news_feats = pd.DataFrame({\n",
        "        \"news_sentiment_1h\": sents_1h,\n",
        "        \"news_sentiment_4h\": sents_4h,\n",
        "        \"news_sentiment_24h\": sents_24h,\n",
        "        \"news_counts_1h\": counts_1h,\n",
        "    })\n",
        "    news_feats[\"news_has_data\"] = ((news_feats[\"news_counts_1h\"] > 0).astype(int))\n",
        "    news_feats.index = left.sort_values(\"index\")[\"index\"].values\n",
        "    return news_feats.reindex(candles.index)\n",
        "\n",
        "print(\"Aligning news to df_all..........\")\n",
        "news_feats_all = align_news_to_candles(df_all)\n",
        "print(f\"rows : {len(news_feats_all)} rows\")\n",
        "print(f\" news_has_data = 1: {int(news_feats_all['news_has_data'].sum())} rows\")\n",
        "print(f\" news_has_data = 0: {int((news_feats_all['news_has_data'] == 0).sum())} rows\")\n",
        "\n",
        "print(\"\\nHead of last 5 candles with news: \")\n",
        "tail_width_news = news_feats_all[news_feats_all[\"news_has_data\"] == 1].tail(5)\n",
        "print(tail_width_news.to_string() if len(tail_width_news) > 0 else \" (none) \")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4dMxH1nR0lNC",
        "outputId": "62482562-cf35-468e-c14c-8fb4beec8ddb"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Aligning news to df_all..........\n",
            "rows : 50000 rows\n",
            " news_has_data = 1: 16 rows\n",
            " news_has_data = 0: 49984 rows\n",
            "\n",
            "Head of last 5 candles with news: \n",
            "       news_sentiment_1h  news_sentiment_4h  news_sentiment_24h  news_counts_1h  news_has_data\n",
            "49986                0.3           0.450000        1.967033e-01               1              1\n",
            "49987               -0.7           0.105556        9.578947e-02               1              1\n",
            "49991               -0.8          -0.800000       -3.373494e-02               1              1\n",
            "49994               -0.1          -0.638462        3.857143e-02               1              1\n",
            "49999                0.3           0.300000       -2.485574e-17               1              1\n"
          ]
        }
      ]
    }
  ]
}