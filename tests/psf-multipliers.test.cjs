const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const multipliers = require('../static/psf-multipliers.js');
const { monthlyPoints, hourlyPoints } = require('../static/forecast.js');
const root = path.join(__dirname, '..');

function response(factor = 55) {
  return {
    success: true, psf_multiplier_product: factor, missing_psfs: [],
    results: Array.from({ length: 8 }, (_, index) => {
      const psf = index + 1;
      const level = [2, 3, 4, 7].includes(psf) ? 1 : 0;
      return {
        psf_id: `F.${psf}`, predicted_label: level, predicted_text: `F.${psf}.${level + 1}`,
        psf_multiplier: psf === 4 ? factor : 1,
        probabilities: psf === 4 ? null : { [`F.${psf}.1`]: 0.7, [`F.${psf}.2`]: 0.3 },
      };
    }),
  };
}

test('products preserve factors below and above 1, without the former HEP discontinuity', () => {
  for (const factor of [0.33, 1, 2, 54, 55, 1000000]) {
    assert.equal(multipliers.calculateProduct(response(factor).results), factor);
    assert.equal(multipliers.validateResponse(response(factor)), factor);
  }
  const reversed = response(); reversed.results.reverse();
  assert.equal(multipliers.validateResponse(reversed), 55);
});

test('incomplete, duplicate, stale and mismatched response data fail instead of defaulting to 1', () => {
  const mutations = [
    data => { data.results.pop(); },
    data => { data.results[7] = { ...data.results[0] }; },
    data => { delete data.results[0].psf_multiplier; },
    data => { delete data.psf_multiplier_product; },
    data => { data.psf_multiplier_product = 1; },
    data => { data.results[0].psf_multiplier = '1'; },
    data => { data.results[0].psf_multiplier = 0; },
    data => { data.results[0].psf_multiplier = NaN; },
    data => { data.results[0].psf_id = 'F.9'; },
  ];
  for (const change of mutations) {
    const data = response(); change(data);
    assert.throws(() => multipliers.validateResponse(data));
  }
});

function pageFixture(filename, data, role) {
  const nodes = {};
  class Element {
    constructor() { this.children = []; this.style = {}; this.textContent = ''; this.innerHTML = ''; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    replaceChildren(...children) { this.children = children; }
  }
  const get = id => nodes[id] ||= new Element();
  const ready = [];
  const context = vm.createContext({
    document: { getElementById: get, createElement: () => new Element(), addEventListener: (_, callback) => ready.push(callback) },
    sessionStorage: { getItem: key => key === 'psfResults' ? JSON.stringify(data) : role },
    console: { log() {} },
  });
  vm.runInContext(fs.readFileSync(path.join(root, 'static/psf-multipliers.js'), 'utf8'), context);
  const html = fs.readFileSync(path.join(root, filename), 'utf8');
  for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) vm.runInContext(match[1], context);
  ready.forEach(callback => callback());
  return { nodes, context, html };
}

test('actual summary and detail page scripts display the same multiplier without role coefficients', () => {
  for (const filename of ['result.html', 'detail_result.html']) {
    for (const role of ['기관사', '관제사', '정비사', '선로작업자']) {
      const page = pageFixture(filename, response(), role);
      const text = page.nodes.multiplierResult.children.map(node => node.textContent).join(' ');
      assert.match(text, /55배/);
      assert.doesNotMatch(text, /%/);
      assert.equal(typeof page.context.calculateHEP, 'undefined');
      assert.doesNotMatch(page.html, /diagnosisHEP|executionHEP|finalHEP|roleHEPValues|hep_multiplier/);
      if (filename === 'detail_result.html') {
        const factor = page.nodes.resultsList.children[3].children[0].children[0];
        assert.equal(factor.textContent, 'PSF 배수: 55배');
        const classificationCard = page.nodes.resultsList.children[0].children[0].innerHTML;
        assert.match(classificationCard, /70\.0%/); // Model class probabilities still use percent.
      }
    }
  }
});

test('both result pages reject old stored payloads and request a new submission', () => {
  const stale = response(); delete stale.psf_multiplier_product;
  for (const filename of ['result.html', 'detail_result.html']) {
    const page = pageFixture(filename, stale, '기관사');
    const text = filename === 'result.html' ? page.nodes.multiplierResult.textContent : page.nodes.errorText.textContent;
    assert.match(text, /설문을 다시 제출/);
  }
});

test('monthly and hourly plots use server factors without HEP conversion or an upper cap', () => {
  const data = response(4.692);
  data.results[5].psf_multiplier = 11.576;
  const factors = Array.from({ length: 601 }, (_, index) => index < 10 ? 4.692 : 4.704);
  const monthly = monthlyPoints({ series: { x_start: 0, x_end: 60, x_step: 0.1, psf_multipliers: factors } }, data.results, multipliers.calculateProduct);
  assert.ok(Math.abs(monthly[0].y - 54.314592) < 1e-9);
  assert.ok(Math.abs(monthly[10].y - 54.453504) < 1e-9);
  assert.ok(monthly[10].y > monthly[0].y);
  const hourly = hourlyPoints({ segments: [{ start_elapsed_hours: 0, end_elapsed_hours: null, predicted_label: 2, psf_multiplier: 17.54 }] }, response(1).results, multipliers.calculateProduct);
  assert.equal(hourly[0].y, 17.54);
  assert.throws(() => hourlyPoints({ segments: [{ start_elapsed_hours: 0, end_elapsed_hours: null, predicted_label: 2 }] }, response().results, multipliers.calculateProduct));
});
