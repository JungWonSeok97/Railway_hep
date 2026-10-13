(function () {
  'use strict';

  function withPrediction(results, psfId, changes) {
    return results.map(result => result.psf_id === psfId ? { ...result, ...changes } : { ...result });
  }

  function hepValue(results, calculate) {
    const value = calculate(results).finalHEP;
    if (!Number.isFinite(value) || value < 0) throw new Error('HEP 계산 결과가 올바르지 않습니다.');
    return value;
  }

  function monthlyPoints(response, results, calculate) {
    const series = response.series;
    if (!series || series.x_start !== 0 || series.x_end !== 60 || series.x_step !== 0.1 ||
        !Array.isArray(series.hep_multipliers) || series.hep_multipliers.length !== 601) {
      throw new Error('월별 그래프 데이터 형식이 올바르지 않습니다.');
    }
    return series.hep_multipliers.map((multiplier, index) => {
      if (!Number.isFinite(multiplier) || multiplier < 0) throw new Error('교육·훈련 배수 데이터가 올바르지 않습니다.');
      return {
        x: Number((series.x_start + index * series.x_step).toFixed(1)),
        y: hepValue(withPrediction(results, 'F.4', { hep_multiplier: multiplier }), calculate),
        multiplier,
      };
    });
  }

  function hourlyPoints(response, results, calculate, horizon = 24) {
    const segments = response.segments;
    if (!Array.isArray(segments) || !segments.length || segments[0].start_elapsed_hours !== 0) {
      throw new Error('시간별 그래프 구간 데이터가 올바르지 않습니다.');
    }
    const points = [];
    segments.forEach((segment, index) => {
      const start = segment.start_elapsed_hours;
      const end = segment.end_elapsed_hours;
      if (!Number.isFinite(start) || start < 0 || ![0, 1, 2].includes(segment.predicted_label) ||
          (end !== null && (!Number.isFinite(end) || end < start)) ||
          (index < segments.length - 1 && end !== segments[index + 1].start_elapsed_hours) ||
          (index === segments.length - 1 && end !== null)) {
        throw new Error('시간별 그래프 구간 데이터가 올바르지 않습니다.');
      }
      if (start >= horizon) return;
      const y = hepValue(withPrediction(results, 'F.1', {
        predicted_label: segment.predicted_label,
        predicted_text: `F.1.${segment.predicted_label + 1}`,
      }), calculate);
      points.push({ x: start, y, level: segment.predicted_label });
      points.push({ x: Math.min(end === null ? horizon : end, horizon), y, level: segment.predicted_label });
    });
    return points;
  }

  function kstValue(value) {
    const time = Date.parse(value);
    if (!Number.isFinite(time)) return '';
    return new Date(time + 9 * 3600000).toISOString().slice(0, 16);
  }

  const core = { monthlyPoints, hourlyPoints, kstValue };
  if (typeof module !== 'undefined' && module.exports) module.exports = core;
  if (typeof document === 'undefined') return;

  const labels = ['보통', '나쁨', '매우나쁨'];
  const number = value => Number(value.toPrecision(6)).toLocaleString('ko-KR', { maximumSignificantDigits: 6 });
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function svgNode(tag, attrs, text) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function drawChart(target, points, kind) {
    target.replaceChildren();
    const monthly = kind === 'monthly';
    const limit = monthly ? 60 : 24;
    const unit = monthly ? '개월' : '시간';
    const title = monthly ? '향후 60개월 HEP 변화' : '조회 시점부터 24시간 HEP 변화';
    const values = points.map(point => point.y);
    const maxY = Math.max(...values) * 1.12 || 1;
    const plot = { left: 90, top: 38, width: 620, height: 230 };
    const x = value => plot.left + value / limit * plot.width;
    const y = value => plot.top + plot.height - value / maxY * plot.height;
    const svg = svgNode('svg', { viewBox: '0 0 750 335', role: 'img', 'aria-label': title, class: 'forecast-svg' });
    svg.append(svgNode('title', {}, title));
    svg.append(svgNode('desc', {}, '가로축은 조회 시점부터 경과 시간, 세로축은 HEP입니다. 아래 슬라이더로 수치를 확인할 수 있습니다.'));
    for (let i = 0; i <= 4; i++) {
      const value = maxY * i / 4;
      svg.append(svgNode('line', { x1: plot.left, x2: x(limit), y1: y(value), y2: y(value), class: 'forecast-grid-line' }));
      svg.append(svgNode('text', { x: plot.left - 12, y: y(value) + 4, 'text-anchor': 'end' }, Number(value.toPrecision(3)).toString()));
    }
    for (let i = 0; i <= 6; i++) {
      const value = limit * i / 6;
      svg.append(svgNode('text', { x: x(value), y: 292, 'text-anchor': 'middle' }, value));
    }
    svg.append(svgNode('text', { x: plot.left, y: 20 }, 'HEP'));
    svg.append(svgNode('text', { x: 400, y: 322, 'text-anchor': 'middle' }, `조회 시점부터 경과 ${unit}`));
    svg.append(svgNode('polyline', {
      points: points.map(point => `${x(point.x)},${y(point.y)}`).join(' '),
      fill: 'none', stroke: monthly ? '#0f3d91' : '#0b6a88', 'stroke-width': 3,
      'vector-effect': 'non-scaling-stroke',
    }));
    const marker = svgNode('circle', { r: 5, fill: '#fff', stroke: '#0f3d91', 'stroke-width': 3 });
    svg.append(marker);
    const scroller = node('div', undefined, 'forecast-chart-scroll');
    scroller.append(svg);
    const sliderId = `${kind}ForecastPosition`;
    const label = node('label', '확인할 시점');
    label.htmlFor = sliderId;
    const slider = node('input');
    slider.type = 'range'; slider.id = sliderId; slider.min = '0';
    slider.max = String(monthly ? points.length - 1 : 2400); slider.step = '1'; slider.value = '0';
    const output = node('output', '', 'forecast-value');
    output.setAttribute('for', sliderId);
    function inspect() {
      const position = Number(slider.value);
      let point;
      let selectedX;
      if (monthly) { point = points[position]; selectedX = point.x; }
      else {
        selectedX = position / 100;
        // At a boundary, the new interval owns the value (step-after).
        point = points[0];
        for (const candidate of points) if (candidate.x <= selectedX) point = candidate;
      }
      marker.setAttribute('cx', x(selectedX)); marker.setAttribute('cy', y(point.y));
      const extra = monthly ? `교육·훈련 배수 ${number(point.multiplier)}` : `작업부하 ${labels[point.level]}`;
      output.textContent = `${number(selectedX)}${unit} 후 · HEP ${number(point.y)} · ${extra}`;
      slider.setAttribute('aria-valuetext', output.textContent);
    }
    slider.addEventListener('input', inspect);
    target.append(node('h4', title), scroller, label, slider, output);
    inspect();
  }

  async function post(url, payload) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload), signal: controller.signal,
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || '그래프 데이터를 가져오지 못했습니다.');
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('응답이 지연되고 있습니다. 다시 조회해 주세요.');
      throw error;
    } finally { clearTimeout(timeout); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const kinds = ['monthly', 'hourly'];
    const openButtons = kinds.map(kind => document.getElementById(`${kind}ForecastOpen`));
    if (openButtons.some(button => !button)) return;
    let data;
    try { data = JSON.parse(sessionStorage.getItem('psfResults')); } catch { return; }
    if (!data?.success || !Array.isArray(data.results) || data.results.length !== 8 ||
        !Array.from({ length: 8 }, (_, i) => `F.${i + 1}`).every(id => data.results.some(result => result.psf_id === id))) {
      openButtons.forEach(button => { button.disabled = true; });
      return;
    }
    const results = data.results;
    const monthsInput = document.getElementById('forecastMonths');
    const workInput = document.getElementById('forecastWorkStartedAt');
    const saved = data.forecast_inputs || {};
    if (Number.isFinite(saved.training_elapsed_months) && saved.training_elapsed_months >= 0) monthsInput.value = saved.training_elapsed_months;
    if (saved.work_started_at) workInput.value = kstValue(saved.work_started_at);
    const basePsf1 = results.find(result => result.psf_id === 'F.1').predicted_label;
    const opened = new Set();

    async function load(kind) {
      const monthly = kind === 'monthly';
      const form = document.getElementById(monthly ? 'monthlyForecastForm' : 'hourlyForecastForm');
      if (form.dataset.busy === 'true') return;
      const status = document.getElementById(`${kind}Status`);
      const target = document.getElementById(`${kind}Chart`);
      status.classList.remove('forecast-error');
      target.replaceChildren();
      let payload;
      if (monthly) {
        const months = monthsInput.valueAsNumber;
        if (!Number.isFinite(months) || months < 0) {
          status.textContent = '교육 후 경과 개월 수를 입력하고 월별 그래프 조회를 눌러 주세요.';
          return;
        }
        payload = { training_elapsed_months: months };
      } else {
        const start = workInput.value ? `${workInput.value}+09:00` : '';
        if (!start || !Number.isFinite(Date.parse(start)) || Date.parse(start) > Date.now()) {
          status.textContent = '현재 시각 이전의 근무 시작 일시를 입력하고 시간별 그래프 조회를 눌러 주세요.';
          return;
        }
        payload = { work_started_at: start, base_psf1: basePsf1 };
      }
      const controls = [...form.querySelectorAll('input, button')];
      form.dataset.busy = 'true'; controls.forEach(control => { control.disabled = true; });
      target.setAttribute('aria-busy', 'true');
      const loadingImage = node('img'); loadingImage.src = '/images/loading.gif'; loadingImage.alt = '';
      loadingImage.className = 'forecast-loading-image';
      status.replaceChildren(loadingImage, document.createTextNode('그래프를 불러오고 있습니다…'));
      try {
        const response = await post(monthly ? '/api/psf4/forecast' : '/api/psf1/predict', payload);
        const points = monthly ? monthlyPoints(response, results, calculateHEP) : hourlyPoints(response, results, calculateHEP);
        drawChart(target, points, kind);
        status.textContent = monthly
          ? `교육 후 ${number(payload.training_elapsed_months)}개월 기준 · 향후 60개월, 0.1개월 간격의 601개 예측값`
          : `조회 기준: ${kstValue(response.generated_at).replace('T', ' ')} (한국시간) · 근무 시작 후 ${number(response.elapsed_hours)}시간`;
        if (!monthly) {
          const details = node('details');
          details.append(node('summary', '작업부하 변화 구간 확인'));
          const table = node('table', undefined, 'forecast-table');
          const head = node('thead'); const row = node('tr');
          ['조회 시점부터', '작업부하 수준'].forEach(text => row.append(node('th', text)));
          head.append(row); table.append(head);
          const body = node('tbody');
          response.segments.forEach(segment => {
            const tr = node('tr');
            tr.append(node('td', `${number(segment.start_elapsed_hours)}시간 후 ~ ${segment.end_elapsed_hours === null ? '이후 계속' : `${number(segment.end_elapsed_hours)}시간 후`}`), node('td', labels[segment.predicted_label]));
            body.append(tr);
          });
          table.append(body); details.append(table); target.append(details);
          target.append(node('p', '그래프는 조회 시점부터 24시간을 표시합니다. 마지막 구간은 이후에도 같은 수준으로 유지됩니다.', 'forecast-note'));
        }
      } catch (error) {
        target.replaceChildren();
        status.textContent = `그래프 조회 실패: ${error.message} 입력값을 확인하고 다시 조회해 주세요.`;
        status.classList.add('forecast-error');
      } finally {
        form.dataset.busy = 'false'; controls.forEach(control => { control.disabled = false; });
        target.setAttribute('aria-busy', 'false');
      }
    }
    document.getElementById('monthlyForecastForm').addEventListener('submit', event => { event.preventDefault(); load('monthly'); });
    document.getElementById('hourlyForecastForm').addEventListener('submit', event => { event.preventDefault(); load('hourly'); });
    kinds.forEach(kind => {
      const dialog = document.getElementById(`${kind}ForecastDialog`);
      const openButton = document.getElementById(`${kind}ForecastOpen`);
      openButton.addEventListener('click', () => {
        if (dialog.open) return;
        dialog.showModal();
        document.body.classList.add('forecast-modal-open');
        if (!opened.has(kind)) { opened.add(kind); load(kind); }
      });
      document.getElementById(`${kind}ForecastClose`).addEventListener('click', () => dialog.close());
      dialog.addEventListener('click', event => {
        const rect = dialog.getBoundingClientRect();
        if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right ||
            event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
      });
      // Native dialog handles Escape and keeps keyboard focus inside the popup.
      dialog.addEventListener('close', () => {
        document.body.classList.remove('forecast-modal-open');
        openButton.focus();
      });
    });
  });
})();
