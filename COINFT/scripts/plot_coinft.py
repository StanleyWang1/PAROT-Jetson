#!/usr/bin/env python3
"""
Plot CoinFT FT-only CSV

Usage:
  python3 plot_coinft.py coinft_ft_YYYYMMDD_HHMMSS.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_coinft.py path/to/coinft_ft_*.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)

    # time axis
    if "t_rel" in df.columns:
        t = df["t_rel"].astype(float)
        xlabel = "Time (s) [t_rel]"
    else:
        t = df["t_wall"].astype(float)
        xlabel = "Time (s) [t_wall]"

    # Force
    plt.figure()
    plt.plot(t, df["Fx"].astype(float), label="Fx")
    plt.plot(t, df["Fy"].astype(float), label="Fy")
    plt.plot(t, df["Fz"].astype(float), label="Fz")
    plt.xlabel(xlabel)
    plt.ylabel("Force")
    plt.title("CoinFT Force")
    plt.grid(True)
    plt.legend()

    # Moment/Torque
    plt.figure()
    plt.plot(t, df["Mx"].astype(float), label="Mx")
    plt.plot(t, df["My"].astype(float), label="My")
    plt.plot(t, df["Mz"].astype(float), label="Mz")
    plt.xlabel(xlabel)
    plt.ylabel("Moment")
    plt.title("CoinFT Moment")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
