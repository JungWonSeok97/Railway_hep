const test = require('node:test');
const assert = require('node:assert/strict');
const { evaluate, split } = require('../static/date-time.js');
const now = Date.parse('2026-11-19T15:30:00+09:00');
const parts = { date: '2026-11-18', period: 'AM', hour: '12', minute: '00' };

test('midnight and noon use the correct 24-hour value in Korea time', () => {
  assert.equal(evaluate(parts, true, now).value, '2026-11-18T00:00');
  assert.equal(evaluate({ ...parts, period: 'PM' }, true, now).value, '2026-11-18T12:00');
  assert.equal(evaluate({ ...parts, period: 'PM', hour: '11', minute: '59' }, true, now).value, '2026-11-18T23:59');
});

test('optional empty is allowed; partial, impossible and future dates are rejected', () => {
  assert.equal(evaluate({}, false, now).error, '');
  assert.ok(evaluate({}, true, now).error);
  assert.ok(evaluate({ ...parts, minute: '' }, false, now).error);
  assert.ok(evaluate({ ...parts, date: '2026-02-29' }, true, now).error);
  assert.equal(evaluate({ ...parts, date: '2024-02-29' }, true, now).error, '');
  assert.ok(evaluate({ ...parts, hour: '0' }, true, now).error);
  assert.ok(evaluate({ ...parts, minute: '60' }, true, now).error);
  assert.equal(evaluate({ date: '2026-11-19', period: 'PM', hour: '3', minute: '30' }, true, now).error, '');
  assert.ok(evaluate({ date: '2026-11-19', period: 'PM', hour: '3', minute: '31' }, true, now).error);
});

test('existing drafts and result metadata restore every hour and minute without changing time', () => {
  for (let hour = 0; hour < 24; hour++) {
    for (let minute = 0; minute < 60; minute++) {
      const value = `2026-11-18T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
      assert.deepEqual(evaluate(split(value), true, now), { value, error: '' });
    }
  }
  assert.deepEqual(split('invalid'), { date: '', period: '', hour: '', minute: '' });
});
