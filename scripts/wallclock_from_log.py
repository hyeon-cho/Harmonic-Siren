"""Table 11 helper: extract (epoch, cumulative wall-clock seconds, PSNR, BER) from a
run's save/<exp>/log.txt and report time-to-quality checkpoints.

The trainer logs lines like
    [06-21 08:08:12] Epoch 80, train: loss=..., valid: psnr=40.27, ssim=..., lpips=..., ber=0.22, n_inf=0, 8.6s 56.4s/10.7m
where the trailing "<cum>/<eta>" field carries the cumulative training time.
PSNR logged as 0 with ber=0 is the trainer's "exact reconstruction" sentinel (PSNR = inf).

Usage:
    python scripts/wallclock_from_log.py save/<exp>/log.txt [--csv out.csv] [--at-psnr 40 50]
"""
import argparse
import csv
import re
import sys

LINE = re.compile(
    r"Epoch (\d+),.*?psnr=([0-9.]+),.*?ber=([0-9.]+),.*?\s([0-9.]+)([smh])/([0-9.]+)([smh])\s*$"
)
UNIT = {"s": 1.0, "m": 60.0, "h": 3600.0}


def parse_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            epoch, psnr, ber = int(m.group(1)), float(m.group(2)), float(m.group(3))
            cum = float(m.group(4)) * UNIT[m.group(5)]
            exact = (ber == 0.0 and psnr == 0.0)  # trainer sentinel for BER==0
            rows.append({"epoch": epoch, "seconds": cum,
                         "psnr": float("inf") if exact else psnr, "ber": ber})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to save/<exp>/log.txt")
    ap.add_argument("--csv", default=None, help="write full (epoch,seconds,psnr,ber) CSV here")
    ap.add_argument("--at-psnr", type=float, nargs="*", default=[],
                    help="report first epoch/time reaching each PSNR threshold")
    args = ap.parse_args()

    rows = parse_log(args.log)
    if not rows:
        sys.exit(f"no metric lines found in {args.log}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "seconds", "psnr", "ber"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")

    last = rows[-1]
    print(f"final: epoch={last['epoch']} time={last['seconds']:.1f}s psnr={last['psnr']} ber={last['ber']}")

    exact = next((r for r in rows if r["psnr"] == float("inf")), None)
    if exact:
        print(f"exact reconstruction (BER=0): epoch={exact['epoch']} time={exact['seconds']:.1f}s")

    for thr in args.at_psnr:
        hit = next((r for r in rows if r["psnr"] >= thr), None)
        if hit:
            print(f"PSNR>={thr}: epoch={hit['epoch']} time={hit['seconds']:.1f}s (psnr={hit['psnr']})")
        else:
            print(f"PSNR>={thr}: not reached")


if __name__ == "__main__":
    main()
