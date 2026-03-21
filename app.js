// TsionVision — app.js (v3, 4-channel model)

(() => {
  const dropZone     = document.getElementById('drop-zone');
  const fileInput    = document.getElementById('file-input');
  const dropInner    = document.getElementById('drop-inner');
  const dropSelected = document.getElementById('drop-selected');
  const fileNameDisp = document.getElementById('file-name-display');
  const fileSizeDisp = document.getElementById('file-size-display');
  const fileClear    = document.getElementById('file-clear');
  const consentCheck = document.getElementById('consent-check');
  const btnRun       = document.getElementById('btn-run');

  const resultSection = document.getElementById('result-section');
  const errorSection  = document.getElementById('error-section');
  const errorMessage  = document.getElementById('error-message');

  const resultSubtitle = document.getElementById('result-subtitle');
  const resultBadge    = document.getElementById('result-badge');
  const staticImg      = document.getElementById('static-img');
  const gifImg         = document.getElementById('gif-img');
  const staticCaption  = document.getElementById('static-caption');
  const gifCaption     = document.getElementById('gif-caption');
  const warningBox     = document.getElementById('warning-box');
  const warningText    = document.getElementById('warning-text');
  const metaGrid       = document.getElementById('meta-grid');
  const metaBlock      = document.getElementById('meta-block');
  const scanIdDisplay  = document.getElementById('scan-id-display');
  const channelBadge   = document.getElementById('channel-badge');

  let selectedFile = null;

  function formatBytes(bytes) {
    if (bytes < 1024)        return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function setFile(file) {
    selectedFile = file;
    fileNameDisp.textContent = file.name;
    fileSizeDisp.textContent = formatBytes(file.size);
    dropInner.style.display = 'none';
    dropSelected.classList.add('visible');
    updateSubmitBtn();
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    dropInner.style.display = '';
    dropSelected.classList.remove('visible');
    updateSubmitBtn();
  }

  function updateSubmitBtn() {
    btnRun.disabled = !(selectedFile && consentCheck.checked);
  }

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) setFile(fileInput.files[0]);
  });

  fileClear.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
  });

  consentCheck.addEventListener('change', updateSubmitBtn);

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const allowed = ['.dcm', '.dicom', '.nii', '.nii.gz', '.zip'];
    const ok = allowed.some(ext => file.name.toLowerCase().endsWith(ext));
    if (!ok) { showError('Unsupported file type. Upload a .dcm, .nii.gz, or .zip'); return; }
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    setFile(file);
  });

  btnRun.addEventListener('click', async () => {
    if (!selectedFile) return showError('Please select a file before submitting.');
    if (!consentCheck.checked) return showError('Please confirm the consent statement.');

    btnRun.classList.add('loading');
    btnRun.disabled = true;
    hideResult();
    hideError();

    const formData = new FormData();
    formData.append('dicom_file', selectedFile);
    // clinical_notes added post-submission via PATCH

    try {
      const res  = await fetch('/api/submit-scan', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok) {
        if (data.header) renderMeta(data.header);
        showError(data.error || 'Server error. Please try again.');
        return;
      }
      renderResult(data);
    } catch (err) {
      showError('Could not reach the server. Is it running?');
    } finally {
      btnRun.classList.remove('loading');
      btnRun.disabled = false;
    }
  });

  function renderResult(data) {
    const { scan_id, header, static_b64, gif_b64, slice_idx, channels, warning, error } = data;

    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    currentScanId = scan_id;
    scanIdDisplay.textContent = `Scan ID: ${scan_id}`;
    setNotesDisplay(null);  // reset notes on new result
    channelBadge.textContent  = `${channels || '?'}/4 channels`;
    renderMeta(header);

    // Warning (fewer than 4 channels)
    if (warning) {
      warningBox.classList.remove('hidden');
      warningText.textContent = warning;
    } else {
      warningBox.classList.add('hidden');
    }

    if (static_b64) {
      staticImg.src = `data:image/png;base64,${static_b64}`;
      staticImg.style.display = 'block';
      staticCaption.textContent =
        `Axial slice ${slice_idx} · Left: FLAIR anatomy · Centre: probability map · Right: overlay (threshold > 0.3)`;
      resultSubtitle.textContent = `Analysis complete · Most active slice: ${slice_idx}`;
      resultBadge.textContent    = 'Complete';
    } else {
      staticImg.style.display = 'none';
      resultSubtitle.textContent = 'Scan stored · Inference unavailable';
      resultBadge.textContent    = 'Stored';
      staticCaption.textContent  = error || 'Model not loaded.';
    }

    if (gif_b64) {
      gifImg.src = `data:image/gif;base64,${gif_b64}`;
      gifImg.style.display = 'block';
      gifCaption.textContent = 'Animated sweep through most active slices · Tumour regions highlighted in red-yellow';
    } else {
      gifImg.style.display = 'none';
      gifCaption.textContent = '';
    }
  }

  function renderMeta(header) {
    if (!metaGrid) return;
    metaGrid.innerHTML = '';
    if (!header || Object.keys(header).length === 0) {
      if (metaBlock) metaBlock.style.display = 'none';
      return;
    }
    const fields = {
      'Patient ID':   header.patient_id,
      'Patient Name': header.patient_name,
      'Date of Birth':header.patient_dob,
      'Sex':          header.patient_sex,
      'Study Date':   header.study_date,
      'Modality':     header.modality,
      'Institution':  header.institution,
    };
    Object.entries(fields).forEach(([key, val]) => {
      if (!val) return;
      const item = document.createElement('div');
      item.className = 'meta-item';
      item.innerHTML = `<div class="meta-key">${key}</div><div class="meta-val">${val}</div>`;
      metaGrid.appendChild(item);
    });
    if (metaGrid.children.length % 2 !== 0) {
      metaGrid.appendChild(Object.assign(document.createElement('div'), { className: 'meta-item' }));
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorSection.classList.remove('hidden');
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    hideResult();
  }

  function hideError()  { errorSection.classList.add('hidden'); }
  function hideResult() { resultSection.classList.add('hidden'); }

  // ── Clinical notes editor (post-submission) ───────────────────────────────
  let currentScanId = null;

  const notesEditBtn  = document.getElementById('notes-edit-btn');
  const notesDisplay  = document.getElementById('notes-display');
  const notesEditor   = document.getElementById('notes-editor');
  const notesTextarea = document.getElementById('notes-textarea');
  const notesCancel   = document.getElementById('notes-cancel');
  const notesSave     = document.getElementById('notes-save');

  function showNotesEditor() {
    notesEditor.classList.remove('hidden');
    notesEditBtn.style.display = 'none';
    notesTextarea.focus();
  }

  function hideNotesEditor() {
    notesEditor.classList.add('hidden');
    notesEditBtn.style.display = '';
  }

  function setNotesDisplay(text) {
    if (text && text.trim()) {
      notesDisplay.innerHTML = `<p class="notes-text">${text.replace(/\n/g, '<br>')}</p>`;
    } else {
      notesDisplay.innerHTML = '<span class="notes-empty">No clinical notes added.</span>';
    }
  }

  notesEditBtn.addEventListener('click', showNotesEditor);
  notesCancel.addEventListener('click', hideNotesEditor);

  notesSave.addEventListener('click', async () => {
    if (!currentScanId) return;
    const notes = notesTextarea.value.trim();
    notesSave.textContent = 'Saving…';
    notesSave.disabled = true;
    try {
      const res = await fetch(`/api/scans/${currentScanId}/notes`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes }),
      });
      if (res.ok) {
        setNotesDisplay(notes);
        notesTextarea.value = '';
        hideNotesEditor();
      }
    } catch (e) {
      console.error('Failed to save notes', e);
    } finally {
      notesSave.textContent = 'Save Note';
      notesSave.disabled = false;
    }
  });

  // Store scan id when result renders
  const _origRenderResult = renderResult;
})();
