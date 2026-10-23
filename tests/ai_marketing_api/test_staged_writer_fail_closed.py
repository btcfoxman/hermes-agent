"""Real staged writer failures stop before another paid editorial surface."""
from __future__ import annotations

import base64
from copy import deepcopy
import json

import httpx
import pytest

import ai_marketing_api.main as api
from ai_marketing_api.operator_private_receipts import PRIVATE_GENERATION_CAPTURE
from ai_marketing_api.operator_receipts import request_sha256
from ai_marketing_api.operator_steps import artifact_signature
from tests.ai_marketing_api.test_editorial_steps_and_scorecard import fact_first, payload
from tests.ai_marketing_api.test_operator_api import _request
from tests.ai_marketing_api.test_private_generation_receipts import install_provider, model_body, provider_reply, InterruptedBody


BASE = '/api/v1/operators/industry/editorial-steps/'


@pytest.fixture(autouse=True)
def isolated_edge(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_API_KEY', 'operator-test-secret')
    monkeypatch.setenv('HERMES_OPENAI_API_KEY', 'provider-private-secret')
    monkeypatch.setenv('HERMES_OPENAI_BASE_URL', 'https://model.example/v1')
    monkeypatch.setenv('HERMES_MODEL', 'offline-writer')
    monkeypatch.setattr(api, '_resolve_host_addresses', lambda host, port: {'93.184.216.34'})


def assert_unavailable(data, *, code='writer_model_not_completed'):
    assert data['status'] == 'unavailable'
    assert data['error_code'] == code
    assert data['artifact'] is None and 'artifact_signature' not in data
    assert data['publishable'] is False and data['requires_human_review'] is True
    assert data['generation_trace']['stop_reason'] == code
    assert PRIVATE_GENERATION_CAPTURE.get() is None


@pytest.mark.parametrize('bad_content', [
    '', '[]', 'PRIVATE_INVALID_JSON {',
    '```json\n{"master_title":"Official release","blocks":[{"kind":"opinion","text":"Use predicate to estimate "material ready".","evidence_ids":[]}],"platform_variants":[]}\n```',
])
def test_real_http_returned_invalid_writer_json_is_not_a_draft_and_cached_without_rebuy(monkeypatch, bad_content):
    calls = install_provider(monkeypatch, lambda request: provider_reply(bad_content))
    body = {'compose':payload(), 'budget_reservation_id':'writer_bad_json'}
    first = _request('POST', BASE+'draft_master', json=body)
    assert first.status_code == 200
    data = first.json()
    assert_unavailable(data)
    trace = data['generation_trace']
    assert trace['origin'] == 'deterministic_fallback' and trace['model_calls'] == 1
    assert trace['attempts'][0]['outcome'] == 'failed'
    event = trace['provider_requests'][0]
    assert event['status'] == 'returned' and event['http_status'] == 200
    assert event['error_code'] == 'provider_response_invalid' and event['usage']['prompt_tokens'] == 120
    receipt = data['private_generation_receipt']
    assert receipt['provider_responses'][0]['content']['value'] == bad_content
    assert receipt['writer_candidate']['source'] == 'fallback'
    assert _request('POST', BASE+'draft_master', json=body).json() == data
    restored = _request('GET', BASE+'draft_master/receipts/'+body['budget_reservation_id'],
        headers={'X-Hermes-Request-SHA256':request_sha256(body)}).json()
    assert restored['status'] == 'completed' and restored['response'] == data
    assert len(calls) == 1


@pytest.mark.parametrize('partial', [False, True])
def test_unknown_provider_outcome_remains_unknown_when_writer_is_unavailable(monkeypatch, partial):
    def provider(request):
        if partial:
            return httpx.Response(200, stream=InterruptedBody(), headers={'X-Oneapi-Request-Id':'partial-original'})
        return httpx.Response(200, content=b'{INVALID_PROVIDER_ENVELOPE')
    calls = install_provider(monkeypatch, provider)
    data = _request('POST', BASE+'draft_master', json={'compose':payload()}).json()
    assert_unavailable(data)
    event = data['generation_trace']['provider_requests'][0]
    assert event['status'] == 'unknown'
    if not partial:
        assert event['http_status'] == 200
    assert event['error_code'] == ('provider_timeout' if partial else 'provider_response_invalid')
    row = data['private_generation_receipt']['provider_responses'][0]
    assert row['response_received'] is True and row['http_status'] == 200
    if partial:
        assert row['body']['complete'] is False and row['body']['hash_scope'] == 'observed_prefix'
    else:
        assert base64.b64decode(row['body']['value']) == b'{INVALID_PROVIDER_ENVELOPE'
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['empty', 'metadata', 'title_only', 'empty_blocks', 'block_metadata', 'extra_surface',
    'wrong_block_type', 'wrong_kind_type', 'wrong_variants_type'])
def test_successful_http_json_still_requires_a_real_master_shape(monkeypatch, kind):
    def provider(request):
        valid = fact_first(model_body(request))
        candidate = {'empty':{}, 'metadata':{'schema_version':'operator.content.v1'},
            'title_only':{'master_title':valid['master_title']},
            'empty_blocks':{**valid, 'blocks':[]}, 'block_metadata':{**valid, 'blocks':[{'metadata':'not a block'}]},
            'wrong_block_type':{**valid, 'blocks':[123]}, 'wrong_kind_type':{**valid, 'blocks':[{'kind':[], 'text':'not a legal block'}]},
            'wrong_variants_type':{**valid, 'platform_variants':{}},
            'extra_surface':{**valid, 'platform_variants':[{'platform':'wechat_mp','blocks':valid['blocks']} ]}}[kind]
        return provider_reply(json.dumps(candidate))
    calls = install_provider(monkeypatch, provider)
    data = _request('POST', BASE+'draft_master', json={'compose':payload()}).json()
    assert_unavailable(data)
    assert data['generation_trace']['origin'] == 'model'
    assert data['generation_trace']['attempts'][0]['outcome'] == 'returned'
    assert data['generation_trace']['provider_requests'][0]['status'] == 'returned'
    assert data['private_generation_receipt']['writer_candidate']['source'] == 'model'
    assert len(calls) == 1


def master_from_provider(monkeypatch):
    calls = install_provider(monkeypatch, lambda request: provider_reply(json.dumps(fact_first(model_body(request)))))
    result = _request('POST', BASE+'draft_master', json={'compose':payload()}).json()
    assert result['status'] == 'artifact'
    return result, calls


@pytest.mark.parametrize('kind', ['empty', 'metadata', 'missing_target', 'empty_blocks', 'extra_surface', 'invalid_json'])
def test_adapt_requires_exact_target_shape_and_failure_is_not_an_artifact(monkeypatch, kind):
    master, _ = master_from_provider(monkeypatch)
    def provider(request):
        valid = fact_first(model_body(request))
        target = valid['platform_variants'][0]
        candidate = {'empty':{}, 'metadata':{'note':'not an adaptation'},
            'missing_target':{'platform_variants':[{**target,'platform':'wechat_moments'}]},
            'empty_blocks':{'platform_variants':[{**target,'blocks':[]}]},
            'extra_surface':{'platform_variants':[target,{**target,'platform':'wechat_moments'}]}}.get(kind)
        return provider_reply('INVALID {' if kind == 'invalid_json' else json.dumps(candidate))
    calls = install_provider(monkeypatch, provider)
    body = {'compose':payload(), 'surface':'wechat_mp', 'candidate':master['artifact'],
        'candidate_signature':master['artifact_signature'], 'budget_reservation_id':'adapt_invalid'}
    data = _request('POST', BASE+'adapt_platform', json=body).json()
    assert_unavailable(data)
    assert _request('POST', BASE+'adapt_platform', json=body).json() == data
    assert len(calls) == 1


@pytest.mark.parametrize('trace_kind', ['missing', 'failed', 'fallback'])
def test_old_signed_fallback_master_cannot_start_an_adapt_provider_call(monkeypatch, trace_kind):
    master, _ = master_from_provider(monkeypatch)
    old = deepcopy(master['artifact'])
    if trace_kind == 'missing':
        old['generation_trace'] = None
    elif trace_kind == 'failed':
        old['generation_trace']['attempts'][0].update(outcome='failed', error_code='provider_response_invalid')
    else:
        old['model'] = 'fallback'
        old['generation_trace']['origin'] = 'deterministic_fallback'
    calls = install_provider(monkeypatch, lambda request: pytest.fail('A technical fallback cannot buy an adaptation'))
    data = _request('POST', BASE+'adapt_platform', json={'compose':payload(), 'surface':'wechat_mp',
        'candidate':old, 'candidate_signature':artifact_signature(old)}).json()
    assert_unavailable(data, code='master_writer_not_completed')
    assert data['generation_trace']['model_calls'] == 0 and not calls


def test_stage_specific_prompts_preserve_feedback_and_canonical_master_with_single_target(monkeypatch):
    original = payload()
    original['editorial_feedback'] = 'Revise the master argument; later adapt the platform format. Preserve this exact feedback.'
    seen = []
    def provider(request):
        envelope = json.loads(request.content)
        body = model_body(request)
        seen.append((envelope, body))
        candidate = fact_first(body)
        if body['channels']:
            # The staged contract requests only this variant, no rewritten master.
            candidate = {'platform_variants':candidate['platform_variants']}
        return provider_reply(json.dumps(candidate))
    calls = install_provider(monkeypatch, provider)
    master = _request('POST', BASE+'draft_master', json={'compose':original}).json()
    assert master['status'] == 'artifact'  # canonical-only is a valid writer, not a fabricated high score
    adapted = _request('POST', BASE+'adapt_platform', json={'compose':original,'surface':'wechat_mp',
        'candidate':master['artifact'],'candidate_signature':master['artifact_signature']}).json()
    assert adapted['status'] == 'artifact'
    for field in ('master_title','master_content','blocks'):
        assert adapted['artifact'][field] == master['artifact'][field]
    assert [variant['platform'] for variant in adapted['artifact']['platform_variants']] == ['wechat_mp']
    assert master['artifact']['quality_scorecard'] is None and adapted['artifact']['quality_scorecard'] is None
    for envelope, body in seen:
        assert envelope['response_format'] == {'type':'json_object'}
        assert 'one strict JSON object' in envelope['messages'][0]['content']
        assert 'Do not wrap it in Markdown fences' in envelope['messages'][0]['content']
        assert 'unescaped ASCII double quotes' in envelope['messages'][0]['content']
        assert body['editorial_feedback'] == original['editorial_feedback']
    master_contract = seen[0][1]['response_contract']
    assert master_contract['platform_variants'] == []
    assert seen[0][1]['stage_writer_contract']['permitted_output_surfaces'] == ['master']
    adapt_contract = seen[1][1]['response_contract']
    assert [variant['platform'] for variant in adapt_contract['platform_variants']] == ['wechat_mp']
    assert adapt_contract['platform_variants'][0]['format'] == 'article'
    assert 'master_title' not in adapt_contract and 'blocks' not in adapt_contract
    assert seen[1][1]['master_draft'] == {'title':master['artifact']['master_title'],'body':master['artifact']['master_content']}
    assert len(calls) == 2
