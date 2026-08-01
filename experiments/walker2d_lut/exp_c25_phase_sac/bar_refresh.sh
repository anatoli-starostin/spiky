#!/usr/bin/env bash
# exp_c25 — keep the Slack progress bar 01a27c5c carrying LIVE per-arm scores.
#
# Slack has no table markup, so the per-arm block is emitted as a fenced code block:
# a pipe-and-dash markdown table renders as literal pipes there and is unreadable.
# Monospace is the only thing that actually lines the columns up in the message body.
#
# Exits when the last trainer disappears, so it cannot outlive the sweep and keep
# posting a frozen number as if it were current.
set -u
cd "$(dirname "$0")"
BAR=01a27c5c
PROG=/home/astarostin/work/slack-facade/progress.py

while true; do
  LINE=""
  PCT=0
  for F in 0 0p85 1p703 2p55; do
    L=$(tail -1 "run_c25_f$F.log" 2>/dev/null)
    IT=$(sed -n "s/^\[ *\([0-9]*\)\/10000\].*/\1/p" <<<"$L")
    RET=$(sed -n "s/.*MJX ret *\([0-9.-]*\).*/\1/p" <<<"$L")
    COV=$(sed -n "s/.*row-cov *\([0-9.]*\)%.*/\1/p" <<<"$L")
    BEST=$(sed -n "s/.*best *\([0-9.-]*\).*/\1/p" <<<"$L")
    case $F in
      0)     NAME="f = 0  (control)" ;;
      0p85)  NAME="f = 0.85 Hz  0.5x" ;;
      1p703) NAME="f = 1.703 Hz 1.0x" ;;
      2p55)  NAME="f = 2.55 Hz  1.5x" ;;
    esac
    LINE+=$(printf "%-18s %8s %8s %7s%%\n" "$NAME" "${RET:--}" "${BEST:--}" "${COV:--}")
    LINE+=$'\n'
    [ -n "${IT:-}" ] && [ "${IT:-0}" -gt "$PCT" ] && PCT=$IT
  done
  P=$((PCT / 100))

  STATS="exp_c25 phase-aware SAC LUT, 4 arms concurrent, 10k iters each, seed 4, c21 hyperparameters.
Live MJX proxy at iter ${PCT}/10000:
\`\`\`
arm                 MJX ret     best  row-cov
$LINE\`\`\`
Early-training ordering is the noise floor, not a result: c21 read 425 at iter 1500 and finished at 5287. The number that decides this is the 100-episode deterministic CPU eval each arm gets at the end."

  python3 "$PROG" update "$BAR" --pct "$P" --stats "$STATS" >/dev/null 2>&1

  pgrep -f "phase_lut_sac.py --addressing" >/dev/null || break
  sleep 120
done
echo "trainers gone; bar refresher exiting at $(date -u +%FT%TZ)"
