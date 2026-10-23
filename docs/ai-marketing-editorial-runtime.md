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

An explicit narrow exception applies only to `industry` with
`genre=industry_analysis`: the exact standalone identifier `Seedance 2.5` may
remain in both source and proposed expression without its version digits alone
making that ordinary fact ineligible. Changed or omitted identifiers, additional
numbers, quantities, dates, currency, prices, fees and commitments retain the
existing exact-wording rule. A unit-adjacent value such as `Seedance 2.5元` is not
a model identifier. This exception only permits an independent support check;
it never sets verified flags or bypasses that check. Unknown, contradicted or
unavailable verification still renders the canonical source.

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

## Optional paragraph layout

Canonical reference objects and authored blocks may include `paragraph_id`.
An ID starts with an ASCII letter and contains at most 48 ASCII letters, digits,
underscores or hyphens. Invalid IDs reject the candidate. Only consecutive blocks
with the same nonempty ID render as one natural paragraph, joined by a single
space. Different IDs and absent IDs retain the existing double newline. A reused
nonconsecutive ID does not reorder or merge separated blocks.

Layout changes neither canonical text, references, required/locked flags nor
evidence binding hashes. Every required fact remains present. Layout is covered
by the whole-artifact signature and its rendered result by the editorial content
hash, so changing it needs fresh review. With no ID, serialized blocks omit the
new field and legacy rendered bytes remain unchanged. Existing receipts are not
rewritten or re-signed.

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

## Genre, style references and scored review

Propose, revise and compose accept `genre` (an optional lowercase identifier)
and `style_references: [{reference_id, title?, url?, genre?, style_notes?, excerpt?}]`.
They travel as a separate `style_context`, including through focused repairs
and independent editing. They are untrusted expression examples, never factual
evidence, product capabilities or the owner's biography. Reference facts must
not be added to `claims` or `authorized_context`.

Compose also accepts `editorial_feedback` (at most 6000 characters) for a prior
independent editor's revision diagnostics. Unlike raw `constraints`, which are
intentionally withheld from the external model, this explicit field reaches
drafting, adaptation, focused repairs and independent review as untrusted data.
It never enters claims or authorized context. Known non-public excerpts are
rejected before model dispatch. The orchestrator remains responsible for
projecting safe review diagnostics rather than raw private source material.

The personal operator has an explicit `genre=public_tutorial` mode for public
tool tutorials. It can propose and compose without owner cards from approved
`company_public` facts or primary/official `industry` facts explicitly
authorized for `personal_ip`. Canonical claims remain fact-only, and the
proposal must have `first_person=false`. It rejects personal attribution,
invented hands-on testing, owner/client results and mixed identity, experience
or owner-opinion claims. Source screenshots and generated illustrations never
establish an experiment by the owner. Missing public evidence still yields
`needs_input`; other genres retain the original personal-card requirements.

For a request with genre or style references, the editor must return
`quality_scorecard.v1`. Each `surfaces` row identifies `master` or an exact
requested platform and contains integer 1–5 scores for `reader_relevance`,
`information_gain`, `analytical_depth`, `natural_expression`, and `platform_fit`,
plus explicit `major_issues` and optional actionable `issues`. Every surface
must occur exactly once. All five scores must be at least 4 and major issues
must be empty; means and author-supplied pass flags are insufficient.
The four boolean review checks must also pass and `failed_surfaces` must be
empty. Ordinary actionable `issues` remain visible at both review and surface
level and do not turn a qualifying scored review into a rejection. Legacy
reviews without a scorecard still require an empty `issues` list.

The server stamps `passed` and `content_sha256`. The hash covers the UTF-8,
sorted-key, compact JSON projection `{master_title, master_content,
platform_variants:[{platform,title,body}]}`, preserving variant order. It is a
rendered-copy hash, not orchestration's package/asset approval hash.

Genre-aware structure allows a sourced event or figure to open a data brief
and does not impose an authored-hook position or fixed number of editorial
blocks. Independent scoring judges genre-appropriate substance. Evidence,
numeric, personal-card, privacy and disclosure gates still apply. Safe server
templates do not become authored content. Legacy requests without these new
fields retain the boolean-review contract. Scored review uses a dedicated,
stable independent-editor system prompt rather than the writing prompt.

## Internal single-call editorial steps

`POST /api/v1/operators/{role_id}/editorial-steps/{step}` uses the existing
service Bearer credential. Supported steps are `draft_master`, `adapt_platform`,
`verify_claims`, and `review_editorial`. Body:

```json
{
  "compose": {"objective": "...", "channels": ["wechat_mp"], "genre": "company_data_brief"},
  "candidate": null,
  "candidate_signature": null,
  "surface": "master",
  "budget_reservation_id": "existing-reservation-id"
}
```

`compose` is a complete ordinary compose request, including approved proposal,
claims and authorized evidence; the example above only illustrates the envelope.
Each step performs at most one actual provider HTTP request, never retries a
second address or repairs its own output. Claim verification can complete
without an upstream call when no eligible paraphrase was proposed.

The response is `operator.editorial_step.v1` with
`status=artifact|blocked|unavailable`, `artifact`, `artifact_signature`, and
`generation_trace`. It always has `publishable=false` and
`requires_human_review=true`. An internal artifact's content status cannot
authorize publishing or replace the final evidence and owner-review gates.

1. `draft_master` returns a normalized master artifact with no platform variants.
2. `adapt_platform` requires that original master artifact and signature, plus a
   surface in the original channel set. It returns one platform variant.
3. Orchestration merges the master with each platform variant. The merged
   object is unsigned. `verify_claims` re-normalizes it from current evidence,
   verifies eligible proposed expressions, renders the actual copy and issues
   a fresh whole-artifact signature.
4. `review_editorial` requires that whole artifact and signature. It assesses
   actual rendered copy and returns a scored, signed internal artifact. Its
   embedded generation trace and claim certificates support orchestration's
   existing authenticated-compose validation.

Signatures are HMAC-SHA256 using the service key over sorted-key, compact,
`ensure_ascii=false` UTF-8 JSON of the whole artifact. Carry the returned object
unchanged; do not manufacture signatures or reuse one after merging. Rotation
of the service key invalidates previous signatures and requires revalidation.
Unsigned input can propose public expressions but cannot attest verification.

## Server routes and provider cost evidence

The standalone image reads optional `$HERMES_HOME/config.yaml` (default
`/data/hermes/config.yaml`) without importing Hermes core. Stage routes live in
server configuration:

```yaml
marketing_api:
  stage_routes:
    compose:
      model: writer-deployment
      base_url: https://models.example/v1
      api_key_env: HERMES_OPENAI_API_KEY
    claim_verification:
      model: evidence-deployment
      base_url: https://models.example/v1
      api_key_env: HERMES_OPENAI_API_KEY
    editorial_review:
      model: editor-deployment
      base_url: https://models.example/v1
      api_key_env: HERMES_OPENAI_API_KEY
```

`planner`, `propose`, `revise`, and `repair` can also be pinned independently.
An absent per-stage entry uses the existing server-owned `HERMES_MODEL`,
`HERMES_OPENAI_BASE_URL`, and `HERMES_OPENAI_API_KEY` when all three are present.
For the authorized standalone deployment without a mounted YAML file,
`HERMES_<STAGE>_MODEL` can select a server-owned stage model, for example
`HERMES_COMPOSE_MODEL`, `HERMES_CLAIM_VERIFICATION_MODEL`, and
`HERMES_EDITORIAL_REVIEW_MODEL`. These use the existing server endpoint/key;
an explicit YAML stage route takes precedence, and an invalid stage model
does not silently fall back.
An explicitly configured invalid stage fails closed and never silently uses
the default. New step endpoints reject client model/base-URL/API-key overrides.
Legal paired overrides remain supported on the legacy endpoints when no
explicit stage route replaces them.

Authenticated service requests can send
`X-Hermes-Single-Provider-Request: true` and
`X-Hermes-Budget-Reservation-Id: <existing reservation>`. The first bounds every
model invocation to one provider HTTP attempt. It does not reduce a legacy
multi-call compose RPC to one invocation: reserve its entire configured call
budget. New editorial steps already impose a one-call limit.

`generation_trace.provider_requests` contains actual request UUIDs, stage,
model, reservation ID, request hash, upstream request ID when available, usage
or null, and safe error/status fields. Planner and proposal calls carry the
same trace. A timeout or missing usage is unknown, never a fabricated zero.
The runtime reports no inferred CNY amount. The orchestrator reserves before
the RPC and reconciles these events; uncertain calls retain their reservation.
The optional request observer hook runs before each actual request and after
its result, allowing a later stricter cost service without changing model code.

Every physical request limits the UTF-8 JSON of its messages to 150000 bytes
and its output to 24000 completion tokens (default 12000). A smaller server
`marketing_api.provider_limits.max_input_bytes` may tighten the input bound;
neither server settings nor client input may enlarge the absolute bounds.
Oversized messages return `input_budget_exceeded` before provider dispatch and
produce no provider event. These limits support a conservatively configured
reservation upper bound; they do not establish a provider price or CNY charge.

Test deployment readiness performs GET-only health, role and OpenAPI checks.
It prints that paid model smoke was not executed. Real generation acceptance
must run separately through orchestration's budget reservation ledger.

## Offline evaluation scenarios

`tests/ai_marketing_api/fixtures/editorial_scenarios.v1.json` contains 30
synthetic contract exercises: 10 per role, each split into 8 tuning and 2
held-out cases. Material-gap cases expect abstention. All fixtures are marked
synthetic and provide zero human-approval credit; replace with verified real
materials before claiming real editorial quality or owner acceptance.

```bash
python -m ai_marketing_api.evaluation_runner
python -m ai_marketing_api.evaluation_runner --results saved-results.json --split all
```

The default does not create model responses or contact any service; it reports
`not_run` and `qualified=false`. Saved results are rechecked through the actual
evidence normalizer, deterministic renderer, content hash and per-surface score
gates. An offline pass remains a machine-assisted assessment.

Optional live evaluation requires all of `--live --allow-paid --api-url` and
`--reservations`, whose JSON maps every selected scenario ID to its own existing
budget reservation. It uses the bounded legacy compose contract and records
actual provider traces in `--save-results`. Held-out live runs additionally
require `--unlock-heldout`; they are not a tuning loop. The caller owns ledger
reconciliation and uncertain reservations. No live evaluation is enabled by
default, and this runner never publishes or records owner acceptance.
