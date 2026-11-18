const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { getConditionCode } = require('../static/input-encoding.js');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'input.html'), 'utf8');
const option = (code, text = '선택한 조건') => ({
  value: 'room1', disabled: false, text, dataset: { conditionCode: code },
});

test('uses the selected option code even when display order and text change', () => {
  const selected = option('F.1.S.3.3', '8시간 이상인 경우');
  const other = option('F.1.S.3.1', '6시간 이내인 경우');
  const select = { selectedOptions: [selected], options: [selected, other] };
  assert.equal(getConditionCode(select, 'F.1.S.3'), 'F.1.S.3.3');
  select.options.reverse();
  selected.text = '표시 문구만 변경';
  assert.equal(getConditionCode(select, 'F.1.S.3'), 'F.1.S.3.3');
  assert.equal(selected.value, 'room1');
});

test('placeholder has no code, but a real option without a matching code fails closed', () => {
  assert.equal(getConditionCode({ selectedOptions: [{ value: '', disabled: true, dataset: {} }] }, 'F.1.S.3'), undefined);
  assert.equal(getConditionCode({ selectedOptions: [] }, 'F.1.S.3'), undefined);
  for (const code of [undefined, '', 'F.2.S.1.1', 'F.1.S.3', 'F.1.S.3.x', 'F.1.S.3.0']) {
    assert.throws(() => getConditionCode({ selectedOptions: [option(code)] }, 'F.1.S.3'), undefined, String(code));
  }
});

test('browser exposes the same encoding helper without CommonJS', () => {
  const context = vm.createContext({ window: {} });
  vm.runInContext(fs.readFileSync(path.join(root, 'static/input-encoding.js'), 'utf8'), context);
  const helper = context.window.PsfInputEncoding || context.PsfInputEncoding;
  assert.equal(typeof helper?.getConditionCode, 'function');
  assert.equal(helper.getConditionCode({ selectedOptions: [option('F.6.S.9.2')] }, 'F.6.S.9'), 'F.6.S.9.2');
});

function attribute(source, name) {
  const match = source.match(new RegExp(`\\b${name}\\s*=\\s*(["'])(.*?)\\1`, 's'));
  return match && match[2];
}

test('all 157 HTML options carry unique codes belonging to their submitted field', () => {
  const houses = html.split(/<div\b[^>]*class=["']house["'][^>]*>/).slice(1);
  assert.equal(houses.length, 8);
  const expectedCounts = [5, 7, 6, 7, 6, 14, 5, 7];
  const codes = new Set();
  let fieldCount = 0;
  let optionCount = 0;
  houses.forEach((house, psf) => {
    const selects = [...house.matchAll(/<select\b([^>]*)>([\s\S]*?)<\/select>/g)]
      .filter(match => (attribute(match[1], 'class') || '').split(/\s+/).includes('room-select'));
    assert.equal(selects.length, expectedCounts[psf]);
    selects.forEach((select, question) => {
      const field = `F.${psf + 1}.S.${question + 1}`;
      fieldCount++;
      const options = [...select[2].matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)];
      const placeholders = options.filter(match => attribute(match[1], 'value') === '');
      assert.equal(placeholders.length, 1, field);
      for (const match of options) {
        const value = attribute(match[1], 'value');
        if (!value || /\bdisabled\b/.test(match[1])) continue;
        const code = attribute(match[1], 'data-condition-code');
        assert.ok(code, `${field}: ${match[2]} must have a code`);
        assert.equal(code.substring(0, code.lastIndexOf('.')), field);
        assert.ok(!codes.has(code), `duplicate ${code}`);
        codes.add(code);
        optionCount++;
        const selected = { value, disabled: false, dataset: { conditionCode: code } };
        assert.equal(getConditionCode({ selectedOptions: [selected] }, field), code);
      }
    });
  });
  assert.equal(fieldCount, 57);
  assert.equal(optionCount, 157);
  assert.match(html, /<script\b[^>]*src=["']\/static\/input-encoding\.js["']/);
  assert.match(html, /PsfInputEncoding\.getConditionCode/);
  assert.doesNotMatch(html, /optionCodeNumbersByField/);
});

function submissionFixture() {
  const groups = html.split(/<div\b[^>]*class=["']house["'][^>]*>/).slice(1).map(house =>
    [...house.matchAll(/<select\b([^>]*)>([\s\S]*?)<\/select>/g)]
      .filter(match => (attribute(match[1], 'class') || '').split(/\s+/).includes('room-select'))
      .map(select => {
        const options = [...select[2].matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)]
          .filter(match => attribute(match[1], 'value') && !/\bdisabled\b/.test(match[1]))
          .map(match => ({
            value: attribute(match[1], 'value'), disabled: false,
            dataset: { conditionCode: attribute(match[1], 'data-condition-code') },
          }));
        return { value: options[0].value, selectedOptions: [options[0]], options };
      }));
  // Exercise the corrected sleep, binary availability, and thermal examples.
  groups[1][0].selectedOptions = [groups[1][0].options[1]];
  groups[1][0].value = groups[1][0].options[1].value;
  groups[5][9].selectedOptions = [groups[5][9].options[1]];
  groups[5][9].value = groups[5][9].options[1].value;
  const nodes = {
    submitBtn: { disabled: false },
    trainingElapsedMonths: { valueAsNumber: 12 },
    workStartedAt: { value: '' },
    loadingOverlay: { style: {} },
  };
  const calls = [];
  const messages = [];
  const stored = {};
  const sandbox = vm.createContext({
    allSelects: groups.flat(), PsfInputEncoding: { getConditionCode },
    document: {
      getElementById: id => nodes[id],
      querySelector: selector => selector === '.role-btn.active' ? {} : null,
      querySelectorAll: selector => {
        assert.equal(selector, '.house');
        return groups.map(selects => ({ querySelectorAll: () => selects }));
      },
    },
    showModal: message => messages.push(message),
    console: { log() {}, error() {} },
    sessionStorage: { setItem: (key, value) => { stored[key] = JSON.parse(value); } },
    window: { location: {} },
    fetch: async (url, request) => {
      calls.push({ url, method: request.method, data: JSON.parse(request.body) });
      return { ok: true, json: async () => ({ success: true, results: Array.from({ length: 8 }, (_, i) => ({ psf_id: `F.${i + 1}` })) }) };
    },
  });
  const begin = html.indexOf('function submitSurvey()');
  const end = html.indexOf('function showModal(', begin);
  assert.ok(begin >= 0 && end > begin);
  vm.runInContext(html.slice(begin, end), sandbox);
  return { sandbox, groups, nodes, calls, messages, stored };
}

test('the actual submit handler sends explicit codes and preserves forecast inputs', async () => {
  const fixture = submissionFixture();
  fixture.sandbox.submitSurvey();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(fixture.calls.length, 1);
  assert.equal(fixture.calls[0].url, '/api/predict');
  assert.equal(fixture.calls[0].method, 'POST');
  const payload = fixture.calls[0].data;
  assert.equal(Object.keys(payload).length, 8);
  let fields = 0;
  fixture.groups.forEach((selects, psf) => selects.forEach((select, index) => {
    const field = `F.${psf + 1}.S.${index + 1}`;
    assert.equal(payload[`F.${psf + 1}`][field], select.selectedOptions[0].dataset.conditionCode);
    fields++;
  }));
  assert.equal(fields, 57);
  assert.equal(payload['F.1']['F.1.S.3'], 'F.1.S.3.3');
  assert.equal(payload['F.2']['F.2.S.1'], 'F.2.S.1.2');
  assert.equal(payload['F.6']['F.6.S.10'], 'F.6.S.10.2');
  assert.equal(payload['F.4'].training_elapsed_months, 12);
  assert.deepEqual(fixture.stored.psfResults.forecast_inputs, { training_elapsed_months: 12, work_started_at: null });
  assert.equal(fixture.sandbox.window.location.href, '/result');
  assert.deepEqual(fixture.messages, []);
});

test('the actual submit handler blocks requests when an option code is absent or belongs to another question', async () => {
  for (const wrongCode of [undefined, 'F.2.S.1.2']) {
    const fixture = submissionFixture();
    fixture.groups[0][2].selectedOptions[0].dataset.conditionCode = wrongCode;
    fixture.sandbox.submitSurvey();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(fixture.calls.length, 0);
    assert.equal(fixture.messages.length, 1);
    assert.ok(fixture.messages[0].length > 0);
    assert.equal(fixture.nodes.submitBtn.disabled, false);
    assert.equal(fixture.nodes.loadingOverlay.style.display, undefined);
    assert.deepEqual(fixture.stored, {});
  }
});
