# Marketing editorial runtime contract

The marketing edge uses immutable role prompts and an explicitly bounded model
loop. It does not run the unrestricted Hermes agent, access business databases,
fetch arbitrary URLs, approve knowledge, or publish. Research material is still
authorized and supplied through orchestration. Neither a successful HTTP status
nor a deterministic regex pass establishes editorial or business quality.

## Compose additions (backward compatible)

`POST /api/v1/operators/{role_id}/compose` retains `operator.content.v1` and
accepts these optional request fields:

```json
{
  "public_editorial_brief": {
    "schema_version": "operator.editorial_brief.v1",
    "role_id": "commercial",
    "objective": "Help video tool developers plan their first integration",
    "audience": "video toolbox and animation workbench developers",
    "reader_value": "Know which integration behavior to validate first",
    "angle": "Start with a small verification task",
    "tone": "practical and specific",
    "cta": "Read the integration documentation",
    "channels": ["wechat_mp"],
    "constraints": ["Do not invent prices or uptime percentages"],
    "product_id": "api",
    "landing_url": "https://aiid.edu.kg"
  },
  "runtime_budget": {
    "max_model_calls": 4,
    "max_surface_revisions": 2,
    "max_elapsed_seconds": 180,
    "max_output_tokens": 12000
  },
  "fact_expression_mode": "grounded_paraphrase"
}
```

The brief is a deliberate public projection. Raw objective, audience and private
records are not automatically promoted to public model input. Its role must
match the endpoint and its channel set must match the request. A brief gives
editorial intent, not evidence for product performance, prices or biography.

Hard ceilings are 8 model invocations, 3 revisions per surface, 600 seconds and
24,000 output tokens per invocation. The runtime reports invocations separately
from known upstream model calls; missing token usage remains null. An exhausted
budget produces a recoverable insufficient-quality state, never a fixed-story
success. Independent editorial review is required for a public brief or an
owner revision; absent or malformed reviewer output cannot approve the draft.

The response adds:

- `generation_trace`: origin, stages/models, prompt hashes, elapsed time, known
  token usage, repair calls, deadline/budget state and review status.
- `editorial_assessment`: audience relevance, usefulness, specificity/clarity,
  platform fit and actionable issues. It is model assistance, not owner consent.
- `claim_binding_reviews`: independently checked candidate public expressions.

Role prompts remain byte-stable during a request. Retries change request data,
not the role system prompt. Valid surfaces are retained while malformed or
shallow surfaces get bounded repairs. Independent editorial feedback can trigger
one further focused revision and a fresh assessment, within the same budget.

## Grounded public expressions

Default `fact_expression_mode` is `exact`. In `grounded_paraphrase`, a model may
propose `{"block_ref":"...","public_text":"..."}` for ordinary facts. The
normalizer first restores the original canonical block. A separate model call
then tests whether the proposed wording preserves the supported meaning.

Version 1 deliberately retains exact wording for prices, numeric facts, dates,
commitments, identity, experiences and unsafe/ambiguous expressions. Failed or
unavailable support checks use the exact source expression and report that
decision. New facts cannot be introduced by classifying them as editorial prose.

An accepted block keeps `text`, `claim_id`, `evidence_ids`, `source_exact` and
`binding_hash` unchanged, adding `public_text` and server-set
`public_text_verified=true`. The actual `master_content` and variant `body` are
rebuilt with the verified expression before independent editorial assessment.
Attestations bind SHA-256 UTF-8 hashes of source and public text, claim and
evidence IDs, verdict and reviewer model. Model-supplied verified flags are
ignored during initial normalization.

Orchestration must consume this only from the authenticated compose result,
validate the hashes, issue its own persistent provenance certificate and repeat
evidence/freshness gates before approval and handoff. Client edits must never
be able to mint, mutate or silently reuse that certificate for changed text.

## Owner revisions

`owner_revision` accepts `instruction` and
`platform_variants: [{platform, title, body}]`. These are explicit editing
intent supplied by orchestration after an owner action. They are not new
approved evidence. Requested channels cannot expand the original scope.

The model is asked to preserve owner wording where supported, but output is
still rebuilt from reviewed structured blocks. The service does not promise
byte-identical preservation of direct edits. Orchestration retains the owner's
candidate on failure and clears old approval when committing a changed draft.

## Editorial planning

`POST /api/v1/operators/editorial/plan` uses the same service Bearer token and
`X-Hermes-OpenAI-*`, model and timeout overrides as role endpoints. Request:

- `mandate`: `objective`, `audience`, `products` (each `product_id`, `name`, `url`,
  `audience`, `value_propositions`), `channels`, `constraints`.
- `opportunities`: `id`, `role_id`, `title`, `summary`, `verified`, `url`.
- `recent_topics`, `weekly_coverage`, `max_packages` and bounded `budget`.

Only verified supplied IDs can be selected, with the original role. Invented,
duplicate, cross-role or excess selections fail closed. The result is
`status=planned|no_opportunities|unavailable`, `selections` (opportunity ID, role,
topic, angle and rationale), role `skips`, model and generation trace. Empty
days are valid; model failure is not silently relabeled as a successful plan.
The orchestrator still owns source verification, duplicate suppression,
authorization, durable state and the choice of an explicitly labeled rule plan.

## Verification

Run `scripts/run_tests.sh tests/ai_marketing_api -- -q`. The wrapper also supports
native Windows virtualenvs launched through Git Bash without exposing provider
credentials to tests. Tests cover unseen subject shapes, planner scope, owner
intent, fallback transparency, global budgets, separate editing assessment,
independent expression support and immutable evidence. Mock-assisted contract
tests do not establish live writing quality, real publishing or user-time gains.

Deployment probes must exercise generic inputs. The compose endpoint no longer
calls the historical funds-remedy or human-review fixed-story builders; their
legacy helper definitions are not a production success path.
