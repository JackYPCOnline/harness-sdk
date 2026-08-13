# Durable Lambda Agent

This deployable example checkpoints Strands model and tool middleware stages with
[AWS Lambda durable execution](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html).
Completed model and tool steps are restored instead of re-executed after a replay.

## Prerequisites

- Node.js 22+
- AWS SAM CLI 1.143+
- AWS CLI v2
- AWS credentials with permission to deploy the template
- Bedrock model access in the deployment region

Use least-privilege credentials in a development or sandbox account. Review the IAM
policies in `template.yml` before deployment; the example uses wildcard resources so
it can work with model IDs and inference profiles across accounts.

## Install and validate

From the monorepo root:

```bash
npm ci
npm --prefix strands-ts/examples/durable-lambda-agent ci
npm --prefix strands-ts/examples/durable-lambda-agent run build
npm --prefix strands-ts/examples/durable-lambda-agent run bundle
npm --prefix strands-ts/examples/durable-lambda-agent run validate:template
```

## Deploy

Deployment creates or updates a CloudFormation stack, IAM role, log group, and durable
Lambda function. SAM displays the change set for confirmation before applying it.

```bash
cd strands-ts/examples/durable-lambda-agent
AWS_REGION=us-west-2 npm run deploy
```

Optional environment variables:

```bash
STACK_NAME=my-durable-agent \
FUNCTION_NAME=my-durable-agent \
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6 \
AWS_REGION=us-west-2 \
npm run deploy
```

## Invoke

Each invocation gets a unique durable execution name.

```bash
npm run invoke          # normal execution
npm run invoke:restart  # wait-driven replay
npm run invoke:crash    # fail and retry the second model step
```

The invocation command follows CloudWatch logs until interrupted with `Ctrl-C`.
For the replay and retry scenarios, each tool log should appear once across the
full durable execution.

## Remove the example stack

Stack deletion removes the function and IAM role and is destructive. Verify the
account, region, and stack name before running it manually:

```bash
aws sts get-caller-identity
aws cloudformation delete-stack \
  --stack-name "${STACK_NAME:-strands-durable-agent}" \
  --region "${AWS_REGION:-us-west-2}"
```

## Scope

This example supports synchronous tools and non-streaming durable replay. Stream
events are emitted during initial execution but are not re-emitted from completed
steps on replay. MCP sessions and Strands interrupts require separate lifecycle
handling and are not covered here.
