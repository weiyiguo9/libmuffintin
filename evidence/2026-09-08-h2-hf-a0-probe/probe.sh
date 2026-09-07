#!/bin/zsh
S=$1; CAP=720
cd /Users/zerozaki07/tmp/libmuffintin
export RAYON_NUM_THREADS=10 DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib"
target/release/examples/h2_hf $S/run --box 8 --orbital-g 4 --field-g 12 --product-g 4 --product-lmax 2 --overlap-tolerance 1e-4 --exchange-coulomb periodic-finite-body --lexp 14 --speed-of-light 137035.9895 --rmt 0.65 > $S/run.log 2>&1 &
pid=$!
echo "pid=$pid start=$(date +%s)" > $S/rss.log
t=0; sampled=0
while kill -0 $pid 2>/dev/null; do
  sleep 20; t=$((t+20))
  rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
  cpu=$(ps -o %cpu= -p $pid 2>/dev/null | tr -d ' ')
  echo "t=${t}s rss_mb=$((rss/1024)) cpu=$cpu" >> $S/rss.log
  if [ $t -ge 240 ] && [ $sampled -eq 0 ]; then sample $pid 15 -mayDie -file $S/sample.txt >/dev/null 2>&1; sampled=1; fi
  if [ $t -ge $CAP ]; then echo "CAP reached, killing" >> $S/rss.log; kill -9 $pid; break; fi
done
wait $pid; echo "exit=$? end=$(date +%s)" >> $S/rss.log
