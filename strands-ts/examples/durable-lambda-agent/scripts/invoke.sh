#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
STACK_NAME="${STACK_NAME:-strands-durable-agent}"
PROMPT="${PROMPT:-Plan my trip to Seattle.}"
SIMULATE_RESTART=false
CRASH_ON_SECOND_CYCLE=false
SCENARIO=baseline

case "${1:-}" in
  --restart)
    SIMULATE_RESTART=true
    SCENARIO=restart
    ;;
  --crash)
    CRASH_ON_SECOND_CYCLE=true
    SCENARIO=crash
    ;;
  "") ;;
  *)
    echo "Usage: $0 [--restart|--crash]" >&2
    exit 2
    ;;
esac

FUNCTION_NAME=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionName'].OutputValue | [0]" \
  --output text)

if [[ -z "$FUNCTION_NAME" || "$FUNCTION_NAME" == "None" ]]; then
  echo "FunctionName output not found for stack $STACK_NAME" >&2
  exit 1
fi

EXECUTION_NAME="strands-${SCENARIO}-$(date +%s)"
PAYLOAD=$(printf '{"prompt":"%s","simulateRestart":%s,"crashOnSecondCycle":%s}' \
  "$PROMPT" "$SIMULATE_RESTART" "$CRASH_ON_SECOND_CYCLE")

echo "Invoking function=$FUNCTION_NAME, region=$REGION, scenario=$SCENARIO"
aws lambda invoke \
  --function-name "${FUNCTION_NAME}:\$LATEST" \
  --invocation-type Event \
  --cli-binary-format raw-in-base64-out \
  --durable-execution-name "$EXECUTION_NAME" \
  --payload "$PAYLOAD" \
  --region "$REGION" \
  /dev/stdout >/dev/null

echo "Invocation queued as $EXECUTION_NAME. Press Ctrl-C to stop following logs."
aws logs tail "/aws/lambda/$FUNCTION_NAME" \
  --region "$REGION" \
  --since 1m \
  --follow
