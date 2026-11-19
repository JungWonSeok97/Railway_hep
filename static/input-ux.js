(function () {
  'use strict';
  const sections = [
    ['작업부하', '근무시간과 휴식, 작업 여유를 살펴봅니다.'],
    ['장비 및 기기', '업무에 사용하는 장비와 지원 시스템의 상태를 살펴봅니다.'],
    ['절차 및 지침', '업무 절차의 명확성과 활용하기 쉬운 정도를 살펴봅니다.'],
    ['교육 및 훈련', '교육 경과시간과 직무에 필요한 경험·지식을 살펴봅니다.'],
    ['의사소통', '정보를 전달하고 동료와 상황을 공유하는 조건을 살펴봅니다.'],
    ['환경', '기상과 실내 환경, 작업공간의 조건을 살펴봅니다.'],
    ['작업절차', '작업계획과 역할 분담, 협업 상태를 살펴봅니다.'],
    ['복잡성', '의사결정에 필요한 정보와 판단의 복잡성을 살펴봅니다.'],
  ];
  const $ = id => document.getElementById(id);
  const panels = Array.from(document.querySelectorAll('.house'));
  const selects = Array.from(document.querySelectorAll('.room-select'));
  const roleButtons = Array.from(document.querySelectorAll('.role-btn'));
  const training = $('trainingElapsedMonths'), started = $('workStartedAt');
  const timePicker = PsfDateTime.init(started);
  const draftKey = 'psfSurveyDraft.v1', total = selects.length + 2;
  let current = 0, initializing = true, validationShown = false, saveTimer;
  let activeDialog = null, returnFocus = null;
  const role = () => document.querySelector('.role-btn.active')?.dataset.role || '';
  const validTraining = () => training.value.trim() !== '' && Number.isFinite(training.valueAsNumber) && training.valueAsNumber >= 0;
  const validStart = () => !timePicker.getError();
  const announce = text => { $('surveyAnnouncement').textContent = text; };
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  panels.forEach((panel, index) => {
    panel.id = `psf-section-${index + 1}`;
    panel.setAttribute('role', 'region');
    panel.setAttribute('aria-label', sections[index][0]);
    panel.querySelector('.house-title').classList.add('visually-hidden');
    panel.querySelectorAll('.room-select').forEach((select, question) => {
      const group = select.closest('.question-group');
      const number = document.createElement('span');
      number.className = 'question-number';
      number.textContent = String(question + 1).padStart(2, '0');
      number.setAttribute('aria-hidden', 'true');
      group.querySelector('.label-text').prepend(number);
      if (Array.from(select.options).some(option => option.textContent.trim().length > 43)) group.classList.add('long-options');
    });
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'section-link';
    button.setAttribute('aria-controls', panel.id);
    button.innerHTML = `<span class="section-number" aria-hidden="true">${String(index + 1).padStart(2, '0')}</span><span class="section-link-text"><span>${sections[index][0]}</span><small class="section-link-count"></small></span><span class="section-check" aria-hidden="true">✓</span>`;
    button.addEventListener('click', () => goTo(index));
    $('sectionNav').appendChild(button);
  });
  const navButtons = Array.from($('sectionNav').children);
  const timeGroup = started.closest('.question-group');
  timeGroup.classList.add('optional-question');
  panels[0].appendChild(timeGroup);
  training.closest('.question-group').classList.add('training-question');

  function getIssues() {
    const issues = [];
    if (!role()) issues.push({ control: roleButtons[0], group: $('roleGroup'), section: 0, message: '직종을 선택해주세요.' });
    panels.forEach((panel, index) => {
      if (index === 3 && !validTraining()) issues.push({ control: training, group: training.closest('.question-group'), section: index, message: '교육 경과 개월 수를 0 이상의 숫자로 입력해주세요.' });
      panel.querySelectorAll('.room-select').forEach(select => {
        if (!select.value) issues.push({ control: select.closest('.question-group').querySelector('input[type="radio"]'), group: select.closest('.question-group'), section: index, message: '현재 상황에 맞는 조건을 하나 선택해주세요.' });
      });
      if (index === 0 && !validStart()) issues.push({ control: timePicker.control, group: timeGroup, section: index, message: timePicker.getError() });
    });
    return issues;
  }
  function clearErrors() {
    document.querySelectorAll('.is-invalid').forEach(group => group.classList.remove('is-invalid'));
    document.querySelectorAll('[aria-invalid="true"]').forEach(control => {
      control.removeAttribute('aria-invalid'); control.removeAttribute('aria-describedby');
    });
    document.querySelectorAll('.field-error').forEach(error => error.remove());
  }
  function renderErrors(issues) {
    clearErrors();
    $('validationSummary').hidden = !issues.length;
    if (!issues.length) return;
    $('validationSummary').textContent = `확인이 필요한 항목이 ${issues.length}개 있습니다. 표시된 항목을 입력한 뒤 결과를 확인해주세요.`;
    issues.forEach((issue, index) => {
      issue.group.classList.add('is-invalid');
      const error = document.createElement('p');
      error.className = 'field-error'; error.id = `survey-error-${index}`; error.textContent = issue.message;
      issue.group.appendChild(error);
      const target = issue.group.querySelector('[role="radiogroup"]') || issue.control;
      target.setAttribute('aria-invalid', 'true'); target.setAttribute('aria-describedby', error.id);
    });
  }
  function persistDraft() {
    saveTimer = null;
    const answers = {};
    selects.forEach(select => {
      const code = select.selectedOptions[0]?.dataset.conditionCode;
      if (code) answers[select.id] = code;
    });
    try {
      sessionStorage.setItem(draftKey, JSON.stringify({ version: 1, answers, role: role(), training: training.value, started: started.value, startedParts: timePicker.getParts(), section: current }));
      $('saveStatus').textContent = '이 탭에 임시 저장되었습니다.';
      $('saveStatus').classList.add('is-saved');
    } catch (_) {
      $('saveStatus').textContent = '임시 저장을 사용할 수 없습니다. 이 페이지에서 작성을 마쳐주세요.';
    }
  }
  function refresh() {
    const answered = selects.filter(select => select.value).length + Number(Boolean(role())) + Number(validTraining());
    const percent = Math.round(answered / total * 100);
    $('progressPercent').innerHTML = `${percent}<span>%</span>`;
    $('progressCount').textContent = `${answered} / ${total} 완료`;
    $('progressFill').style.width = `${percent}%`;
    $('surveyProgress').setAttribute('aria-valuenow', String(answered));
    $('surveyProgress').setAttribute('aria-valuemax', String(total));
    roleButtons.forEach(button => button.setAttribute('aria-pressed', String(button.classList.contains('active'))));
    selects.forEach(select => select.closest('.question-group').classList.toggle('is-answered', Boolean(select.value)));
    panels.forEach((panel, index) => {
      const fields = Array.from(panel.querySelectorAll('.room-select'));
      const count = fields.filter(select => select.value).length + (index === 3 ? Number(validTraining()) : 0);
      const length = fields.length + Number(index === 3), complete = count === length;
      navButtons[index].querySelector('.section-link-count').textContent = `${count} / ${length} 입력`;
      navButtons[index].classList.toggle('is-complete', complete);
      navButtons[index].setAttribute('aria-label', `${index + 1}. ${sections[index][0]}, ${count}개 입력, 총 ${length}개${complete ? ', 입력 완료' : ''}`);
      if (index === current) {
        $('sectionCount').textContent = `${count} / ${length} 완료`;
        $('sectionCount').classList.toggle('is-complete', complete);
        $('stepRemaining').textContent = complete ? '이 영역의 필수 항목을 모두 입력했습니다.' : `이 영역에 ${length - count}개 항목이 남아 있습니다.`;
      }
    });
    const ready = answered === total && validStart();
    $('submitBtn').classList.toggle('active', ready);
    $('submitBtn').hidden = current !== panels.length - 1 && !ready;
    $('nextSection').hidden = current === panels.length - 1 || ready;
    $('findUnanswered').disabled = ready;
    if (validationShown) renderErrors(getIssues());
    if (!initializing) { clearTimeout(saveTimer); saveTimer = setTimeout(persistDraft, 150); }
  }
  function goTo(index, focus = true) {
    current = Math.max(0, Math.min(panels.length - 1, index));
    panels.forEach((panel, i) => { panel.hidden = i !== current; });
    navButtons.forEach((button, i) => {
      button.classList.toggle('is-current', i === current);
      if (i === current) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current');
    });
    $('stepLabel').textContent = `영역 ${String(current + 1).padStart(2, '0')} / 08`;
    $('sectionTitle').textContent = sections[current][0];
    $('sectionDescription').textContent = sections[current][1];
    $('stepPosition').textContent = `${current + 1} / 8 영역`;
    $('previousSection').disabled = current === 0;
    $('nextSection').hidden = current === panels.length - 1;
    $('submitBtn').hidden = current !== panels.length - 1;
    refresh();
    if (focus) {
      $('sectionTitle').focus({ preventScroll: true });
      document.querySelector('.section-intro').scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth', block: 'start' });
      announce(`${current + 1}번째 영역, ${sections[current][0]}`);
    }
    if (window.innerWidth <= 900) {
      $('sectionNav').scrollLeft = navButtons[current].offsetLeft - $('sectionNav').offsetLeft - 12;
    }
  }
  function validate() {
    validationShown = true;
    const issues = getIssues(); renderErrors(issues);
    if (!issues.length) return true;
    const first = issues[0];
    goTo(first.section, false);
    first.group.scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth', block: 'center' });
    first.control.focus({ preventScroll: true });
    announce(`${issues.length}개 항목을 확인해주세요. ${first.message}`);
    return false;
  }
  function openDialog(overlay) {
    if (activeDialog === overlay) return;
    returnFocus = document.activeElement; activeDialog = overlay;
    overlay.style.display = 'flex'; overlay.setAttribute('aria-hidden', 'false');
    document.querySelector('.container').inert = true;
    document.querySelector('.app-header').inert = true;
    document.body.classList.add('dialog-open');
    overlay.querySelector('button')?.focus();
  }
  function closeDialog() {
    if (!activeDialog) return;
    activeDialog.style.display = 'none'; activeDialog.classList.remove('is-open'); activeDialog.setAttribute('aria-hidden', 'true');
    activeDialog = null;
    document.querySelector('.container').inert = false; document.querySelector('.app-header').inert = false;
    document.body.classList.remove('dialog-open');
    if (returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
    returnFocus = null;
  }
  function setBusy(busy) {
    document.querySelector('.container').inert = busy;
    document.querySelector('.app-header').inert = busy;
    if (busy) {
      const status = $('loadingOverlay').querySelector('[role="status"]');
      status.tabIndex = -1;
      status.focus();
    } else {
      $('submitBtn').focus({ preventScroll: true });
    }
  }
  window.PsfSurvey = { refresh, validate, openDialog, closeDialog, setBusy, get activeDialog() { return activeDialog; } };
  $('previousSection').addEventListener('click', () => goTo(current - 1));
  $('nextSection').addEventListener('click', () => goTo(current + 1));
  $('findUnanswered').addEventListener('click', validate);
  [training, started].forEach(input => input.addEventListener('input', refresh));
  $('dismissDraft').addEventListener('click', () => { $('draftNotice').hidden = true; });
  $('resetSurvey').addEventListener('click', () => openDialog($('resetOverlay')));
  $('cancelReset').addEventListener('click', closeDialog);
  $('confirmReset').addEventListener('click', () => {
    closeDialog(); clearTimeout(saveTimer); saveTimer = null; initializing = true;
    selects.forEach(select => { select.value = ''; select.dispatchEvent(new Event('change', { bubbles: true })); });
    roleButtons.forEach(button => button.classList.remove('active'));
    training.value = ''; timePicker.setValue(''); validationShown = false;
    clearErrors(); $('validationSummary').hidden = true; $('draftNotice').hidden = true;
    try { sessionStorage.removeItem(draftKey); sessionStorage.removeItem('selectedRole'); } catch (_) { /* Optional persistence. */ }
    goTo(0, false); initializing = false;
    $('saveStatus').textContent = '입력 내용은 이 탭에서 임시 저장됩니다.'; $('saveStatus').classList.remove('is-saved');
    roleButtons[0].focus();
    $('roleGroup').scrollIntoView({ behavior: reducedMotion ? 'instant' : 'smooth', block: 'center' });
    announce('입력 내용을 초기화했습니다.');
  });
  $('resetOverlay').addEventListener('click', event => { if (event.target === $('resetOverlay')) closeDialog(); });
  document.addEventListener('keydown', event => {
    if (!activeDialog) return;
    if (event.key === 'Escape') { event.preventDefault(); closeDialog(); }
    if (event.key === 'Tab') {
      const buttons = Array.from(activeDialog.querySelectorAll('button, [href], input, [tabindex="0"]')).filter(el => !el.disabled && el.getClientRects().length);
      const first = buttons[0], last = buttons[buttons.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
  });
  document.querySelectorAll('.scenario-actions button').forEach(button => button.addEventListener('click', () => {
    refresh(); announce('예시 응답을 불러왔습니다. 직종과 교육 경과 개월 수를 확인해주세요.');
  }));
  try {
    const saved = JSON.parse(sessionStorage.getItem(draftKey));
    if (saved?.version === 1 && saved.answers && typeof saved.answers === 'object') {
      selects.forEach(select => {
        const option = Array.from(select.options).find(item => item.dataset.conditionCode === saved.answers[select.id]);
        if (option?.value) { select.value = option.value; select.dispatchEvent(new Event('change', { bubbles: true })); }
      });
      const savedRole = roleButtons.find(button => button.dataset.role === saved.role);
      if (savedRole) { savedRole.classList.add('active'); sessionStorage.setItem('selectedRole', saved.role); }
      if (typeof saved.training === 'string') training.value = saved.training;
      if (saved.startedParts && typeof saved.startedParts === 'object') timePicker.setParts(saved.startedParts);
      else if (typeof saved.started === 'string') timePicker.setValue(saved.started);
      if (Number.isInteger(saved.section)) current = Math.max(0, Math.min(panels.length - 1, saved.section));
      $('draftNotice').hidden = !(savedRole || selects.some(select => select.value) || training.value || started.value);
    }
  } catch (_) { /* Corrupt or unavailable storage never blocks a new survey. */ }
  goTo(current, false); initializing = false;
  window.addEventListener('pagehide', () => { if (saveTimer) { clearTimeout(saveTimer); persistDraft(); } });
})();
