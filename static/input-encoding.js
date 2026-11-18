(function () {
  'use strict';

  function getConditionCode(select, fieldKey) {
    const option = select?.selectedOptions?.[0];
    if (!option || option.disabled || option.value === '') return undefined;

    const code = option.dataset?.conditionCode;
    const prefix = `${fieldKey}.`;
    if (typeof code !== 'string' || !code.startsWith(prefix) ||
        !/^[1-9]\d*$/.test(code.slice(prefix.length))) {
      throw new Error('설문 선택지 정보를 확인할 수 없습니다. 페이지를 새로고침해 주세요.');
    }
    return code;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { getConditionCode };
  } else {
    window.PsfInputEncoding = { getConditionCode };
  }
})();
