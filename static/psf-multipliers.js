(function (root) {
  'use strict';

  const expectedIds = Array.from({ length: 8 }, (_, index) => `F.${index + 1}`);
  const staleMessage = '최신 PSF 가중치 결과가 필요합니다. 설문을 다시 제출해 주세요.';

  function calculateProduct(results) {
    if (!Array.isArray(results) || results.length !== 8) throw new Error(staleMessage);
    const factors = new Map();
    for (const result of results) {
      if (!result || !expectedIds.includes(result.psf_id) || factors.has(result.psf_id) ||
          !Number.isFinite(result.psf_multiplier) || result.psf_multiplier <= 0) {
        throw new Error(staleMessage);
      }
      factors.set(result.psf_id, result.psf_multiplier);
    }
    const product = expectedIds.reduce((value, id) => value * factors.get(id), 1);
    if (!Number.isFinite(product) || product <= 0) throw new Error('PSF 가중치의 곱이 올바르지 않습니다.');
    return product;
  }

  function validateResponse(data) {
    if (!data || data.success !== true) throw new Error(staleMessage);
    const calculated = calculateProduct(data.results);
    const product = data.psf_multiplier_product;
    if (!Number.isFinite(product) || product <= 0 ||
        Math.abs(calculated - product) > Math.max(calculated, product) * 1e-10) {
      throw new Error(staleMessage);
    }
    return product;
  }

  function formatMultiplier(value) {
    return Number(value.toPrecision(6)).toLocaleString('ko-KR', { maximumSignificantDigits: 6 });
  }

  function renderSummary(target, data) {
    const product = validateResponse(data);
    target.replaceChildren();
    const title = document.createElement('h2');
    title.textContent = 'PSF 가중치의 곱';
    const value = document.createElement('p');
    value.className = 'multiplier-value';
    value.textContent = formatMultiplier(product);
    const description = document.createElement('p');
    description.className = 'multiplier-note';
    description.textContent = '8개 PSF 가중치를 곱한 값입니다. 사고 발생 확률이나 안전·비안전 직무 판정이 아닙니다.';
    target.append(title, value, description);
  }

  const api = { calculateProduct, validateResponse, formatMultiplier, renderSummary };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.PsfMultipliers = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
