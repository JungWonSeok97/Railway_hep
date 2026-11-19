(function (root) {
  'use strict';
  const pad = value => String(value).padStart(2, '0');
  const empty = () => ({ date: '', period: '', hour: '', minute: '' });

  function evaluate(parts, required = false, now = Date.now()) {
    const values = ['date', 'period', 'hour', 'minute'].map(key => parts[key] || '');
    if (values.every(value => value === '') && !required) return { value: '', error: '' };
    if (values.some(value => value === '')) return { value: '', error: '날짜, 오전·오후, 시, 분을 모두 선택해주세요.' };
    const { date, period, hour, minute } = parts;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !['AM', 'PM'].includes(period) ||
        !/^(?:[1-9]|1[0-2])$/.test(hour) || !/^[0-5]\d$/.test(minute)) {
      return { value: '', error: '날짜와 시간을 올바르게 선택해주세요.' };
    }
    const day = new Date(`${date}T00:00:00Z`);
    if (!Number.isFinite(day.getTime()) || day.toISOString().slice(0, 10) !== date) {
      return { value: '', error: '유효한 날짜를 선택해주세요.' };
    }
    const hour24 = Number(hour) % 12 + (period === 'PM' ? 12 : 0);
    const value = `${date}T${pad(hour24)}:${minute}`;
    return { value, error: Date.parse(`${value}+09:00`) > now ? '근무 시작 일시는 현재 시각 이전으로 선택해주세요.' : '' };
  }

  function split(value) {
    const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})$/.exec(value || '');
    if (!match || Number(match[2]) > 23 || Number(match[3]) > 59) return empty();
    return { date: match[1], period: Number(match[2]) < 12 ? 'AM' : 'PM', hour: String(Number(match[2]) % 12 || 12), minute: match[3] };
  }

  function init(input) {
    if (input.dateTimePicker) return input.dateTimePicker;
    const host = document.querySelector(`[data-date-time="${input.id}"]`);
    if (!host) return null;
    const fields = Object.fromEntries(['date', 'period', 'hour', 'minute'].map(key => [key, host.querySelector(`[data-part="${key}"]`)]));
    for (const [key, count, start] of [['hour', 12, 1], ['minute', 60, 0]]) {
      for (let i = start; i < start + count; i++) {
        const option = document.createElement('option');
        option.value = key === 'minute' ? pad(i) : String(i);
        option.textContent = `${pad(i)}${key === 'hour' ? '시' : '분'}`;
        fields[key].appendChild(option);
      }
    }
    const getParts = () => Object.fromEntries(Object.entries(fields).map(([key, field]) => [key, field.value]));
    function sync(notify = true) {
      input.value = evaluate(getParts(), host.dataset.required === 'true').value;
      if (notify) input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    function setParts(parts, notify = false) {
      Object.entries(fields).forEach(([key, field]) => { field.value = typeof parts?.[key] === 'string' ? parts[key] : ''; });
      sync(notify);
    }
    function updateMax() { fields.date.max = new Date(Date.now() + 9 * 3600000).toISOString().slice(0, 10); }
    Object.values(fields).forEach(field => field.addEventListener('input', () => sync()));
    fields.date.addEventListener('focus', updateMax);
    host.querySelector('[data-action="today"]').addEventListener('click', () => { updateMax(); fields.date.value = fields.date.max; sync(); fields.period.focus(); });
    host.querySelector('[data-action="clear"]').addEventListener('click', () => { setParts(empty(), true); fields.date.focus(); });
    const picker = {
      getParts, setParts, setValue: value => setParts(split(value)),
      getError: () => evaluate(getParts(), host.dataset.required === 'true').error,
      focus: () => (Object.values(fields).find(field => !field.value) || fields.date).focus(),
      control: fields.date,
    };
    input.dateTimePicker = picker;
    picker.setValue(input.value); updateMax();
    return picker;
  }
  const api = { evaluate, split, init };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.PsfDateTime = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
