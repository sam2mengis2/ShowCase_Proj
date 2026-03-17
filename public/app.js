const fileInput = document.getElementById('file-input');
const fileChip  = document.getElementById('file-chip');
const chipName  = document.getElementById('file-chip-name');
const dropZone  = document.getElementById('drop-zone');

fileInput.addEventListener('change', () => {
	if (fileInput.files.length > 0) {
		chipName.textContent = fileInput.files[0].name;
		fileChip.classList.add('visible');
	}
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
	e.preventDefault();
	dropZone.classList.remove('drag-over');
	const file = e.dataTransfer.files[0];
	if (file && (file.name.endsWith('.dcm') || file.name.endsWith('.dicom'))) {
		const dt = new DataTransfer();
		dt.items.add(file);
		fileInput.files = dt.files;
		chipName.textContent = file.name;
		fileChip.classList.add('visible');
	}
});

function validateForm(e) {
	const consent = document.getElementById('consent');
	const file    = fileInput.files[0];
	const result  = document.getElementById('result');

	if (!file) {
		e.preventDefault();
		result.innerHTML = '<div class="result-error">Please attach a DICOM file before submitting.</div>';
		return false;
	}

	if (!consent.checked) {
		e.preventDefault();
		result.innerHTML = '<div class="result-error">Please confirm the consent statement before submitting.</div>';
		return false;
	}

	return true;
}
