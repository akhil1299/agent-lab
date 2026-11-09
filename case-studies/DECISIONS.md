# Decisions

What I chose, why, and what I rejected.

Each entry names the alternative, not just the choice. Where an entry has a **switch** line, that is the condition under which the other option becomes correct — the alternatives themselves are implemented in `forks.py`.

## Data and features

### rangeBetween(-600, 0), upper bound 0 not -1
The feature describes this transaction's position in the burst, so `txn_count_last_10m = 2` means "this is the 2nd in ten minutes", which is what makes `velocity_flag = count > 1` correct. If I were predicting a label rather than describing the row, the bound would be -1 to keep the row out of its own window.
**Switch:** the moment this feature feeds a trained model rather than a rule, the row must leave its own window.

### rangeBetween, not rowsBetween
`rangeBetween` measures over the ordering *value* — seconds of event time — so "the last 600 seconds" means what it says. `rowsBetween` counts positions, which would give "the last N transactions" regardless of how far apart they are.
Rejected: a row-count window. It answers a different question.

### Features materialised to Delta instead of .cache()
Spark DataFrames are lazy: every action re-runs the whole plan over 2M rows. Serverless has no cache. Writing `card_features` once makes every agent lookup a cheap key read. That table is the feature store.
**Switch:** in production this becomes an online feature store, because a Delta read is still too slow for a synchronous decision.

### Every LLM call goes through one metered wrapper
`call_llm()` logs step, model, latency and tokens into `CALL_LOG`, so cost and latency per step are measured, not guessed.
Rejected: calling the model directly inside each agent. No single place to measure, route or add retries.

## Agents

### Card dispute triage, not plain transaction scoring
The customer's message is evidence that rules can't read, so the LLM has a real job: read the story, compare it with the data, and explain.
Rejected: an LLM over numeric flags only. An if-statement on `rules_fired` could do the same job.

### The state schema is the handoff contract
Each node accepts and returns a fixed shape. That schema is the handshake. On failure the verifier writes the reason into state and control goes back: a loop with repair, not a chain. Fail closed after 3 attempts.
Rejected: free-text handoff. Unparseable, untestable, no contract to violate.

### Pydantic for LLM output, TypedDict for the state
The LLM is where bad data enters, so its output gets a Pydantic model: the provider enforces the schema (`strict=True`) and Pydantic validates again in Python. The state is written by my own code, so a TypedDict is enough.
Rejected: a hand-written JSON Schema dict for the main path. It sends the same schema but skips validation in Python. Implemented as **fork 01**.
**Switch:** a provider or framework with no Pydantic support, or a schema generated at runtime rather than declared.

### Structured output, not raw call and parse
`with_structured_output(..., strict=True)` makes the provider enforce the shape, so there is no JSON to repair and no parse step that can throw.
Rejected: prompting for JSON and calling `json.loads`. Implemented as **fork 02** to measure how often it actually breaks.
**Switch:** a model or endpoint without structured-output support — then the parse path needs its own retry-with-error-in-prompt loop.

### Field descriptions are per-field instructions
The Planner mislabelled a clear case until the `story_vs_data` description said which way to compare ("the data SUPPORTS / CONTRADICTS the customer's claim") with an example for each value.
Rejected: relying on the label names alone.

### Lists with enums for anything the verifier loops over
`checks` and `policies` are lists whose items must be one of the four policy names. In testing, an unconstrained list produced invented names ("User verification status") that match no flag.
Rejected: free strings. An invented policy name makes the verifier's job impossible — it cannot check a flag that does not exist.

### difficulty is computed by code, not by the LLM
The model judges `story_vs_data`; a rule turns it into easy/hard. Routing must be repeatable and testable.
Rejected: asking the model for difficulty. It adds cost and randomness to a decision a rule can make.

### The label is removed from a copy of txn
`is_fraud_burst` is deleted from a copy before prompting, so the model can't copy the answer, and the state keeps it for evaluation.
Rejected: deleting it from the state itself. The evaluation would lose its ground truth.

### Customer text is fenced as data
The message is wrapped in `<message>` tags and labelled as data from the customer, not instructions. A dispute message is untrusted input from outside the system.
Rejected: interpolating the message directly into the prompt.

### The Planner has no tools
Its inputs are put into the state before it runs: fewer calls, the same input every time, easy to audit.
Rejected: a ReAct planner that fetches its own data. More calls and unbounded loops for data I already know it needs.
**Switch:** when the set of data the planner needs depends on the case and can't be known up front.

### The Analyst re-reads the evidence
The Planner leaves notes (where to look), not evidence. The Analyst must cite flags that fired and quote the customer's words, so it needs `txn` and `dispute_text` itself.
Possible optimisation: send only the flags in `checks` plus amount, merchant and channel, since input tokens are most of the cost.

### Model routing: small for easy first attempts, big otherwise
The big model costs roughly 8× more per call. Easy cases on the first attempt get the small one; hard cases and every retry get the big one.
Rejected: always big (pays for easy cases) and always small (fails the hard ones). The cost-versus-accuracy numbers that justify this get measured in the routing block, not assumed here.

### Verifier is plain Python, not an LLM *(in progress)*
An LLM checking an LLM is a second thing that can be wrong, and when it fails you get no signal about which one broke. Deterministic checks are cheap, fast and auditable, which matters in a regulated process.
Rejected: LLM-as-judge on every verdict. Kept for reason quality only, in evaluation.
