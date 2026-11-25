#!/usr/bin/env bash
# Run inside the official task image, once per fresh container. No inference.
set -eu
control=${1:?nop or oracle}
task=${2:-instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86}
case "$control" in nop|oracle) ;; *) exit 2;; esac
[[ "$task" =~ ^instance_[A-Za-z0-9_.-]+$ ]] || exit 2
cd /app 2>/dev/null || cd /testbed
git rev-parse HEAD
git clone --depth 1 --branch v2.0.0 --filter=blob:none --sparse https://github.com/scaleapi/SWE-bench_Pro-os.git /tmp/sbp
test "$(git -C /tmp/sbp rev-parse HEAD)" = 66f92766bba642462d4bbe5479e83f91f9211862
if [ "$control" = oracle ]; then
  git -C /tmp/sbp sparse-checkout set "v2/tasks/$task/tests" "v2/tasks/$task/solution"
  mkdir -p /solution
  cp -a "/tmp/sbp/v2/tasks/$task/solution/." /solution/
  bash /solution/solve.sh
else
  git -C /tmp/sbp sparse-checkout set "v2/tasks/$task/tests"
fi
mkdir -p /tests /logs/verifier
cp -a "/tmp/sbp/v2/tasks/$task/tests/." /tests/
set +e
bash /tests/test.sh
rc=$?
set -e
echo "VERIFIER_EXIT=$rc"
printf 'OFFICIAL_REWARD='; cat /logs/verifier/reward.txt
test -s /logs/verifier/output.json
python3 -c "import json;from collections import Counter; print('PARSED_TEST_COUNTS',dict(Counter(x.get('status') for x in json.load(open('/logs/verifier/output.json')).get('tests',[]))))"
if [ "$control" = nop ]; then test "$rc" != 0; test "$(cat /logs/verifier/reward.txt)" = 0; else test "$rc" = 0; test "$(cat /logs/verifier/reward.txt)" = 1; fi
