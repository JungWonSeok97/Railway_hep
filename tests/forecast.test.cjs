const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { monthlyPoints, hourlyPoints, kstValue } = require('../static/forecast.js');
const root = path.join(__dirname, '..');
const detailScript = [...fs.readFileSync(path.join(root, 'detail_result.html'), 'utf8').matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]).join('\n');
const context = vm.createContext({
  document: { addEventListener() {} },
  sessionStorage: { getItem: key => key === 'selectedRole' ? '기관사' : null },
  console: { log() {} },
});
vm.runInContext(detailScript, context);
const calculate = results => context.calculateHEP(results);
const results = Array.from({ length: 8 }, (_, i) => ({
  psf_id: `F.${i + 1}`,
  predicted_label: [1, 2, 6].includes(i) ? 1 : 0,
  predicted_text: `F.${i + 1}.${[1, 2, 6].includes(i) ? 2 : 1}`,
  ...(i === 3 ? { hep_multiplier: 0.33, probabilities: null } : {}),
}));
const monthly = { success: true, series: { x_start: 0, x_end: 60, x_step: 0.1, hep_multipliers: Array.from({ length: 601 }, (_, i) => 0.33 + i / 1000) } };
const hourly = { success: true, generated_at: '2026-08-25T15:32:02+09:00', elapsed_hours: 4.53, segments: [
  { start_elapsed_hours: 0, end_elapsed_hours: 1.07, predicted_label: 1 },
  { start_elapsed_hours: 1.07, end_elapsed_hours: 2.87, predicted_label: 0 },
  { start_elapsed_hours: 2.87, end_elapsed_hours: 8.27, predicted_label: 1 },
  { start_elapsed_hours: 8.27, end_elapsed_hours: null, predicted_label: 2 },
] };

test('monthly: 601 points, 0.1-month spacing, only PSF4 changes, original results preserved', () => {
  const before = JSON.stringify(results);
  const points = monthlyPoints(monthly, results, calculate);
  assert.equal(points.length, 601);
  assert.equal(points[0].x, 0); assert.equal(points[1].x, 0.1); assert.equal(points[600].x, 60);
  assert.equal(points[0].y, calculate(results).finalHEP);
  assert.ok(Math.abs(points[600].y - (0.01225 + 0.00615) * 0.93) < 1e-12);
  assert.equal(JSON.stringify(results), before);
});

test('hourly: guide intervals become steps and null end continues to 24 hours', () => {
  const points = hourlyPoints(hourly, results, calculate);
  assert.deepEqual(points.map(point => point.x), [0, 1.07, 1.07, 2.87, 2.87, 8.27, 8.27, 24]);
  assert.deepEqual(points.map(point => point.level), [1, 1, 0, 0, 1, 1, 2, 2]);
  assert.ok(Math.abs(points[0].y - (0.01225 + 0.00615) * 14.45 * 0.33) < 1e-12);
  assert.ok(Math.abs(points.at(-1).y - (0.01225 + 0.00615) * 17.54 * 0.33) < 1e-12);
  const terminal = hourlyPoints({ segments: [{ start_elapsed_hours: 0, end_elapsed_hours: null, predicted_label: 2 }] }, results, calculate);
  assert.equal(terminal.length, 2); assert.equal(terminal[0].y, terminal[1].y);
});

test('invalid responses are rejected and time conversion always uses Korea time', () => {
  assert.throws(() => monthlyPoints({ series: {} }, results, calculate));
  assert.throws(() => monthlyPoints({ series: { ...monthly.series, hep_multipliers: Array(601).fill(null) } }, results, calculate));
  assert.throws(() => hourlyPoints({ segments: [] }, results, calculate));
  assert.throws(() => hourlyPoints({ segments: [{ start_elapsed_hours: 0, end_elapsed_hours: null, predicted_label: 3 }] }, results, calculate));
  assert.equal(kstValue('2026-08-25T02:00:00Z'), '2026-08-25T11:00');
  assert.equal(kstValue('invalid'), '');
});

// Minimal DOM doubles exercise the controller; they do not replace visual browser checks.
function setup(savedInputs, failMonthly = false) {
  class Element {
    constructor() { this.children = []; this.attrs = {}; this.listeners = {}; this.dataset = {}; this.disabled = false; this.value = ''; this.open = false; this.classList = { add() {}, remove() {} }; }
    get valueAsNumber() { return this.value === '' ? NaN : Number(this.value); }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    setAttribute(key, value) { this.attrs[key] = String(value); }
    addEventListener(event, handler) { this.listeners[event] = handler; }
    querySelectorAll() { return this.controls || []; }
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.close?.(); }
    focus() { this.focused = true; }
    getBoundingClientRect() { return { left: 100, right: 900, top: 100, bottom: 700 }; }
  }
  const nodes = {};
  const get = id => nodes[id] ||= new Element();
  get('monthlyForecastForm').controls = [get('forecastMonths'), new Element()];
  get('hourlyForecastForm').controls = [get('forecastWorkStartedAt'), new Element()];
  const calls = [];
  let ready;
  const sandbox = vm.createContext({
    document: { body: new Element(), getElementById: get, createElement: () => new Element(), createElementNS: () => new Element(), createTextNode: text => text, addEventListener: (_, handler) => { ready = handler; } },
    sessionStorage: { getItem: () => JSON.stringify({ success: true, results, forecast_inputs: savedInputs }) },
    calculateHEP: calculate, Date, AbortController, setTimeout, clearTimeout,
    fetch: async (url, options) => {
      calls.push({ url, payload: JSON.parse(options.body) });
      if (failMonthly && url.includes('psf4')) return { ok: false, json: async () => ({ success: false, error: 'test failure' }) };
      return { ok: true, json: async () => url.includes('psf4') ? monthly : hourly };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(root, 'static/forecast.js'), 'utf8'), sandbox);
  ready();
  return { nodes, calls };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('each button opens only its own popup/API; close restores focus and reopen preserves the graph', async () => {
  const { nodes, calls } = setup({ training_elapsed_months: 0, work_started_at: '2026-08-25T11:00:00+09:00' });
  nodes.monthlyForecastOpen.listeners.click(); await settle();
  assert.equal(nodes.monthlyForecastDialog.open, true);
  assert.equal(nodes.hourlyForecastDialog.open, false);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/psf4/forecast');
  assert.equal(calls[0].payload.training_elapsed_months, 0);
  assert.ok(nodes.monthlyChart.children.length > 0);
  nodes.monthlyForecastClose.listeners.click();
  assert.equal(nodes.monthlyForecastDialog.open, false);
  assert.equal(nodes.monthlyForecastOpen.focused, true);
  nodes.hourlyForecastOpen.listeners.click(); await settle();
  assert.equal(nodes.hourlyForecastDialog.open, true);
  assert.equal(nodes.monthlyForecastDialog.open, false);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[1].payload, { work_started_at: '2026-08-25T11:00+09:00', base_psf1: 0 });
  assert.ok(nodes.hourlyChart.children.length > 0);
  nodes.hourlyForecastClose.listeners.click();
  nodes.monthlyForecastOpen.listeners.click(); await settle();
  assert.equal(calls.length, 2);
  const dialog = nodes.monthlyForecastDialog;
  dialog.listeners.click({ target: dialog, clientX: 200, clientY: 200 });
  assert.equal(dialog.open, true);
  dialog.listeners.click({ target: dialog, clientX: 20, clientY: 20 });
  assert.equal(dialog.open, false);
});

test('missing old-session inputs prompt for input; API failure keeps other graph and restores retry controls', async () => {
  const missing = setup({}); missing.nodes.monthlyForecastOpen.listeners.click(); await settle();
  assert.equal(missing.calls.length, 0);
  assert.ok(missing.nodes.monthlyStatus.textContent.includes('입력'));
  const failed = setup({ training_elapsed_months: 15, work_started_at: '2026-08-25T11:00:00+09:00' }, true);
  failed.nodes.monthlyForecastOpen.listeners.click(); await settle();
  failed.nodes.monthlyForecastClose.listeners.click();
  failed.nodes.hourlyForecastOpen.listeners.click(); await settle();
  assert.ok(failed.nodes.monthlyStatus.textContent.includes('test failure'));
  assert.equal(failed.nodes.monthlyChart.children.length, 0);
  assert.ok(failed.nodes.hourlyChart.children.length > 0);
  assert.equal(failed.nodes.forecastMonths.disabled, false);
  failed.nodes.monthlyForecastForm.listeners.submit({ preventDefault() {} }); await settle();
  assert.equal(failed.calls.length, 3);
});
