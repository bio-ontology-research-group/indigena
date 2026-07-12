#!/usr/bin/env bash
# Workstation Resnik sweep. Ibex runs the GPU train+eval and writes each fold's matched
# pool/pairs to data/matched_fold<f>/. For each fold this script waits for that dump on
# Ibex, pulls it, and runs the matched Resnik grid locally (groovy/SLIB on the workstation).
# Usage: run_resnik_sweep_workstation.sh [fold ...]   (default 0..9)
set -o pipefail
cd ~/Git/indigena/hierarchy_test
IBEX=glogin.ibex.kaust.edu.sa
IHT=/ibex/user/zhapacfp/indigena/hierarchy_test
export GROOVY=$HOME/.sdkman/candidates/groovy/current/bin/groovy
FOLDS="${*:-0 1 2 3 4 5 6 7 8 9}"

for f in $FOLDS; do
  echo "[ws-resnik fold $f] waiting for Ibex dump $(date '+%H:%M:%S')"
  until ssh -o BatchMode=yes "$IBEX" "test -f $IHT/data/matched_fold${f}/pairs_human_fold${f}.csv" 2>/dev/null; do sleep 120; done
  mkdir -p data/matched_fold${f} data/gene_disease_folds_unified/fold_${f}
  rsync -az -e "ssh -o BatchMode=yes" "$IBEX:$IHT/data/matched_fold${f}/" data/matched_fold${f}/
  # sync the fold split too, so Resnik's leak filter reads Ibex's exact test_diseases
  rsync -az -e "ssh -o BatchMode=yes" "$IBEX:$IHT/data/gene_disease_folds_unified/fold_${f}/" data/gene_disease_folds_unified/fold_${f}/
  echo "[ws-resnik fold $f] running Resnik $(date '+%H:%M:%S')"
  GROOVY=$GROOVY bash run_resnik_matched.sh $f
done
echo "[ws-resnik] all requested folds done $(date '+%H:%M:%S')"
