# Amazon Bedrock in CloudShell

You will send requests to Bedrock models, compare two models, measure concurrency, test prompt caching, and look at a batch job. Everything runs in the AWS console and its built-in terminal, CloudShell.

**You need three files:** this guide, `run.py`, and `workshop_data.json`.

## 1. Set up CloudShell

1. Sign in to your AWS account. In the Region selector, choose **United States (Oregon)** (`us-west-2`).
2. Click the **CloudShell** icon in the console toolbar. Close the welcome dialog and wait for the `$` prompt.
3. Choose **Actions → Upload file** and upload `run.py`. Repeat for `workshop_data.json`. Both land in your home directory.
4. Paste this block. If a **Safe Paste** dialog appears, choose **Paste**.

```bash
export AWS_DEFAULT_REGION=us-west-2
export AWS_PAGER=""
cd ~
python3 run.py prepare
```

You should see **Ready**.

**What `run.py` is:** a small Python script that runs the exercises. It uses only Python's standard library and calls the AWS CLI already installed in CloudShell. Each exercise (`selection`, `sweep`, `cache`) builds the requests, sends them to Bedrock, scores the answers, and packs a results ZIP with a report. Without `--live` it does not contact AWS. It replays saved responses from `workshop_data.json` instead, so you can run it at no cost.

**What `prepare` does:** it creates a `requests` folder with three files.

`requests/quickstart.json` is the request you send in section 3:

```json
{
  "modelId": "amazon.nova-lite-v1:0",
  "messages": [
    {"role": "user", "content": [{"text": "In one sentence, explain what a GPU is used for in deep learning."}]}
  ],
  "inferenceConfig": {"maxTokens": 32, "temperature": 0, "topP": 0.9}
}
```

`requests/playground-A01.txt` is the prompt for the Playground comparison in section 4. It holds the instructions, followed by the first abstract:

```text
Read the abstract below. Extract the explicitly stated research area, primary method, and dataset name.
...
Abstract:
...
```

`requests/batch_input.jsonl` is the input for the batch job in section 8: 100 requests, one per line.

## 2. Models used

| Exercise | Model ID |
| --- | --- |
| First request, concurrency, caching | `amazon.nova-lite-v1:0` |
| Model comparison (second model) | `us.amazon.nova-micro-v1:0` |
| Batch job | `us.amazon.nova-lite-v1:0` |

An ID that starts with `us.` is a US inference profile: Bedrock may route the request to any supported US Region. An ID without it runs only in Oregon.

## 3. Send your first request

This sends one prompt to Nova Lite through Bedrock's **Converse** API: *"In one sentence, explain what a GPU is used for in deep learning."* Everything is read from `requests/quickstart.json`: the model ID, the prompt, and the settings (at most 32 output tokens, temperature 0, Top P 0.9). Run `cat requests/quickstart.json` to see it. The response is saved to `quickstart-result.json`. This is one real request and costs a fraction of a cent.

```bash
aws bedrock-runtime converse --region us-west-2 \
  --cli-input-json file://requests/quickstart.json \
  --output json > quickstart-result.json
cat quickstart-result.json
```

The answer is in `output.message.content`. Record the **input tokens**, **output tokens**, and **stopReason**. `end_turn` means the model finished; `max_tokens` means it hit the 32-token limit.

Saved result:

```json
{"stopReason":"end_turn","usage":{"inputTokens":15,"outputTokens":27,"totalTokens":42},"metrics":{"latencyMs":423}}
```

## 4. Compare two models

**The task:** read a research abstract and pull out three facts — the research area, the method, and the dataset name — as a JSON object a script could use. You will give the same task to two Amazon models: **Nova Lite** (larger) and **Nova Micro** (smaller, cheaper). The question is which one gets the facts right *and* returns them in the exact format asked for.

**The abstracts:** `workshop_data.json` contains 12 short, made-up research abstracts written for this exercise (alloy defects, fluid flow, ecology, bus delays, and so on). It also has the correct answer for each one, so responses can be scored automatically. You'll try the first abstract by hand, then score all 12.

1. In Bedrock, open **Playground → Chat**. Select **Amazon → Nova Lite 1.0 → On-demand**. Turn on **Compare mode** and select **Nova Micro 1.0 → US Nova Micro** as the second model.
2. For both: max output tokens **96**, temperature **0**, Top P **0.9**.
3. Paste this prompt and choose **Run**:

```text
Read the abstract below. Extract the explicitly stated research area, primary method, and dataset name.
Return exactly one JSON object with these three keys: research_area, method, dataset.
Copy the stated labels and dataset name. If no dataset is specified, use JSON null.
Ignore methods mentioned only as baselines. Do not include Markdown, explanations, or extra keys.

Abstract:
Hairline defects in polished alloy coupons can escape a manual microscope inspection when lighting changes between sessions. In materials science, we trained a convolutional neural network on the AlloyScope-120 dataset to classify cropped images as scratched, pitted, or clear. The test partition held out entire coupons so that nearby crops could not appear on both sides of the split. We inspected disagreements under a second lamp setting and found that edges near the specimen holder were especially troublesome. The experiment uses fictional images and illustrates why specimen-level splits matter.
```

A passing answer is one JSON object with exactly these three keys and nothing else. Markdown fences (```` ```json ````), extra text, or extra keys fail.

```json
{"research_area":"Materials science","method":"Convolutional neural network","dataset":"AlloyScope-120"}
```

Now score both models on all 12 abstracts. `run.py` checks each saved response against the answer key in `workshop_data.json`. It replays saved responses, so it sends nothing to AWS and costs nothing:

```bash
python3 run.py selection --group test-01
```

`--group` is a label of your choice (for example `test-01`). It only names your results files, so you can tell runs apart.

| Saved run (12 abstracts) | Valid JSON | All 3 fields correct |
| --- | --- | --- |
| Nova Lite | 3 / 12 | 3 / 12 |
| Nova Micro | 12 / 12 | 11 / 12 |

**Question:** Nova Lite often had the right labels inside Markdown fences. Why does that still count as a failure?

**Answer:** A program reads these answers, not a person. In the saved run, 9 of Nova Lite's 12 answers looked like this:

````text
```json
{"research_area": "ecology", "method": "random forest", "dataset": "MesaPlots-64"}
```
````

A JSON parser such as Python's `json.loads` rejects this text because it starts with ```` ```json ```` instead of `{`. Any pipeline built on this output would crash or skip the record, even though the facts are correct. You would have to write cleanup code or retry, and you have already paid for the tokens. Following the format is part of getting the answer right.

Add `--live` to send the 24 requests to Bedrock yourself.

## 5. Concurrency: does more parallelism help?

**Predict first:** if you raise the number of requests in flight from 1 to 8, will throughput go up? What happens to the slowest requests?

In Bedrock, open **Quotas** and search for **Nova Lite** to see the request and token limits. All requests in the account count toward these limits.

This sends 1 warm-up plus 24 requests at each of 1, 2, 4, and 8 workers (97 requests). Run it once.

```bash
python3 run.py sweep --live --group test-01
```

| Report column | Meaning |
| --- | --- |
| `success_per_s` | Responses returned per second |
| `p95_client_s` | Latency of the slowest 5% of requests, in seconds |
| `failures`, `retries` | Failed requests; extra attempts after a failure |
| `exact` | Responses with all three fields correct |

Saved results:

| Workers | Returned | Responses/s | p95 latency (s) |
| --- | --- | --- | --- |
| 1 | 24/24 | 0.81 | 1.43 |
| 2 | 24/24 | 0.79 | 2.90 |
| 4 | 24/24 | 0.87 | 5.26 |
| 8 | 24/24 | 0.83 | 10.77 |

Throughput stayed near 0.8/s while tail latency rose about 7×. No failures. Each request starts a new AWS CLI process, so CloudShell's own CPU is part of what you measured. This does not prove Bedrock was at capacity.

## 6. Prompt caching

Four questions are asked about the same long reference manual (a fictional document stored in `workshop_data.json`). The **baseline** arm sends it plainly. The **checkpoint** arm marks it as cacheable:

```json
"system": [
  {"text": "<the long manual>"},
  {"cachePoint": {"type": "default"}}
]
```

This is just the structure; `run.py` fills it in. Run the 8 requests:

```bash
python3 run.py cache --live --group test-01
```

Saved checkpoint-arm results:

| Request | Cache write tokens | Cache read tokens |
| --- | --- | --- |
| Q1 | 3,721 | 0 |
| Q2 | 0 | 3,721 |
| Q3 | 0 | 3,721 |
| Q4 | 0 | 3,721 |

Q1 wrote the cache and Q2 through Q4 read from it. A **read** is what proves reuse; a write alone proves nothing. Median latency was 1.16 s (baseline) vs 1.13 s (checkpoint). That difference is too small to claim a speedup. The cache only lasts five minutes and needs a prefix of at least 1K tokens.

## 7. Download your results

Every run prints a line like **Download this ZIP: /home/cloudshell-user/results/....zip**.

1. Copy that full path.
2. In CloudShell, choose **Actions → Download file** and paste the path.
3. Unzip it on your computer and open **report.html**. You don't need AWS or a notebook to view it.

The ZIP also contains `summary.csv`, `observations.csv` (one row per request), and `raw.jsonl` (full API responses).

When you're done, type `exit`.

## 8. Batch inference

A batch job sends many requests at once without you waiting on each one. You put the requests in a file in S3 (AWS's file storage). Bedrock processes them in the background and writes the answers back to S3. It costs half the on-demand price (see section 9), but it can take minutes to hours. Use it when nobody is waiting for an answer, such as processing a whole collection overnight.

**You need:** an S3 bucket in Oregon, and permission to create an IAM service role. The role is what lets Bedrock read and write that bucket.

### Step 1: Check the input file

`prepare` already wrote `requests/batch_input.jsonl`. It holds the 12 abstracts repeated to make 100 records, because Bedrock requires at least 100 records per job. That makes 100 requests, not 100 independent tests. Each line is one request with a unique `recordId`:

```json
{"recordId":"A01-0001","modelInput":{"messages":[{"role":"user","content":[{"text":"Read the abstract below. ..."}]}],"inferenceConfig":{"maxTokens":96,"temperature":0,"topP":0.9}}}
```

### Step 2: Upload it to S3

Bucket names are shared across all of AWS, so pick a unique one. Skip the `mb` line if the bucket already exists.

```bash
BUCKET=bedrock-batch-yourname-01
aws s3 mb s3://$BUCKET --region us-west-2
aws s3 cp requests/batch_input.jsonl s3://$BUCKET/input/batch_input.jsonl
```

### Step 3: Create the job

In Bedrock, open **Batch inference → Create job** and fill in:

| Field | Value |
| --- | --- |
| Job name | Any unique name, such as `abstracts-test-01` |
| Model | **Nova Lite → US Nova Lite** (`us.amazon.nova-lite-v1:0`) |
| Invocation type | **Converse** (the input file uses the Converse request format) |
| Input data | `s3://<your bucket>/input/batch_input.jsonl` |
| Output data | `s3://<your bucket>/output/` |
| Service access | **Create and use a new service role**, or choose an existing role that can read the input and write the output |
| Timeout | 24 hours (the minimum) |

Choose **Create job**.

### Step 4: Wait for it to finish

The status moves through **Submitted → Validating → Scheduled → InProgress → Completed**. Refresh the console, or check from CloudShell:

```bash
aws bedrock list-model-invocation-jobs --region us-west-2 \
  --query 'invocationJobSummaries[].[jobName,status]' --output table
```

If it ends as **Failed**, open the job to see the message. The cause is usually the role not being able to read the bucket.

### Step 5: Get the results

Bedrock writes to `s3://<your bucket>/output/<job ID>/`. Download everything:

```bash
aws s3 cp s3://$BUCKET/output/ batch-results/ --recursive
```

You get two files:

- `batch_input.jsonl.out`: one line per record, with its `recordId`, the input, and either `modelOutput` or an `error`.
- `manifest.json.out`: totals for the job, including records processed, successes, errors, and tokens used.

The output lines may come back in a different order from the input. Match each answer to its question by `recordId`.

### Saved results

A saved run of this job returned **100 responses** with zero errors in **539 s** (0.185 responses/s). Only **25** were exactly correct. "Completed" means the job finished. It does not mean the answers are right.

## 9. Cost per usable answer

Cost = input tokens × input price / 1M + output tokens × output price / 1M. Use the token counts the API returned, not the max-token limit.

| Price snapshot (2026-11-19) | Input / 1M tokens | Output / 1M tokens |
| --- | --- | --- |
| Nova Lite | $0.060 | $0.240 |
| Nova Micro | $0.035 | $0.140 |
| Nova Lite batch | $0.030 | $0.120 |

Divide the total by the number of **exact** answers:

| Saved comparison | Exact | USD per exact answer |
| --- | --- | --- |
| Nova Lite | 3 / 12 | 0.0000772 |
| Nova Micro | 11 / 12 | 0.0000120 |

**Question:** why doesn't a cheap token price guarantee a cheap usable answer?

## 10. Make a recommendation

1. **Scenario:** interactive help (someone waits for each answer) or overnight coding of many abstracts?
2. **Choice:** which model, how many workers or batch, and is caching useful?
3. **Quality:** valid JSON / total, exact / total, and one example failure.
4. **Performance:** throughput, p95 latency or batch time, request count, failures.
5. **Cost:** estimated cost per exact answer.
6. **Next test:** what could change your mind?

> "For ___, we choose ___ because ___. Our evidence is ___ from ___ requests. The main limitation is ___. Next we would test ___."

## 11. If something goes wrong

| Problem | Fix |
| --- | --- |
| `run.py` or `workshop_data.json` not found | Run `cd ~`, then `ls`. Re-upload any missing file. |
| Prompt shows `>` | Paste was incomplete. Press Ctrl+C and paste the whole block again. |
| Output opens in a pager | Press `q`, then run `export AWS_PAGER=""`. |
| Expired credentials | Refresh the console sign-in and reopen CloudShell. |
| AccessDeniedException / ValidationException | Check that your account has access to the model, and check the model ID and Region. |
| ThrottlingException | Keep the partial results. Pause before trying again. |
| No cache reads | Rerun within five minutes. Keep the same model and prefix. |
| Download fails | Paste the entire `.zip` path exactly as printed. |
