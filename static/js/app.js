/**
 * Audio Noise Remover — Frontend Application (v2)
 */

document.addEventListener('DOMContentLoaded', () => {

    // =========================================
    // Particle Background
    // =========================================

    const canvas = document.getElementById('particles-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let particles = [];
        let animationId;

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }

        function createParticles() {
            particles = [];
            const count = Math.floor((canvas.width * canvas.height) / 18000);
            for (let i = 0; i < count; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.3,
                    vy: (Math.random() - 0.5) * 0.3,
                    radius: Math.random() * 1.5 + 0.5,
                    opacity: Math.random() * 0.3 + 0.1,
                });
            }
        }

        function drawParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0) p.x = canvas.width;
                if (p.x > canvas.width) p.x = 0;
                if (p.y < 0) p.y = canvas.height;
                if (p.y > canvas.height) p.y = 0;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(129, 140, 248, ${p.opacity})`;
                ctx.fill();
            });

            // Draw connections
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(99, 102, 241, ${0.06 * (1 - dist / 120)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }

            animationId = requestAnimationFrame(drawParticles);
        }

        resizeCanvas();
        createParticles();
        drawParticles();

        window.addEventListener('resize', () => {
            resizeCanvas();
            createParticles();
        });
    }

    // =========================================
    // DOM Elements
    // =========================================

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('audio-file');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file');
    const uploadForm = document.getElementById('upload-form');
    const processBtn = document.getElementById('process-btn');
    const toggleParamsBtn = document.getElementById('toggle-params');
    const paramsGrid = document.getElementById('params-grid');

    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');

    const progressFill = document.getElementById('progress-fill');
    const progressPercent = document.getElementById('progress-percent');
    const progressMessage = document.getElementById('progress-message');

    const statsGrid = document.getElementById('stats-grid');
    const downloadBtn = document.getElementById('download-btn');
    const newFileBtn = document.getElementById('new-file-btn');
    const retryBtn = document.getElementById('retry-btn');

    let currentJobId = null;
    let selectedFile = null;

    // =========================================
    // Socket.IO
    // =========================================

    const socket = io();

    socket.on('progress', (data) => {
        if (data.job_id === currentJobId) {
            updateProgress(data.percent, data.message);
        }
    });

    socket.on('processing_complete', (data) => {
        if (data.job_id === currentJobId) {
            showResults(data.results);
        }
    });

    socket.on('processing_error', (data) => {
        if (data.job_id === currentJobId) {
            showError(data.error);
        }
    });

    // =========================================
    // Drag & Drop
    // =========================================

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', () => {
        clearFile();
    });

    // =========================================
    // Parameter Toggle
    // =========================================

    toggleParamsBtn.addEventListener('click', () => {
        const isVisible = paramsGrid.style.display !== 'none';
        paramsGrid.style.display = isVisible ? 'none' : 'grid';
        toggleParamsBtn.classList.toggle('active');
    });

    // =========================================
    // Form Submission
    // =========================================

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('audio_file', selectedFile);
        formData.append('ar_order', document.getElementById('ar-order').value);
        formData.append('eta', document.getElementById('eta').value);
        formData.append('forgetting_factor', document.getElementById('forgetting-factor').value);

        processBtn.disabled = true;
        showSection('progress');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                currentJobId = data.job_id;
                updateProgress(5, 'Upload complete. Processing started...');
            } else {
                showError(data.error || 'Upload failed. Please try again.');
            }
        } catch (err) {
            showError('Network error. Please check your connection and try again.');
        }
    });

    // =========================================
    // Navigation
    // =========================================

    newFileBtn.addEventListener('click', resetApp);
    retryBtn.addEventListener('click', resetApp);

    // =========================================
    // Plot Tabs
    // =========================================

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;

            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.plot-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`plot-${tab}`).classList.add('active');
        });
    });

    // =========================================
    // Helper Functions
    // =========================================

    function handleFile(file) {
        const allowedExts = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'opus'];
        const ext = file.name.split('.').pop().toLowerCase();

        if (!allowedExts.includes(ext)) {
            showError(`Unsupported format: .${ext}\nAllowed formats: ${allowedExts.join(', ')}`);
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        dropZone.style.display = 'none';
        processBtn.disabled = false;
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
        dropZone.style.display = 'block';
        processBtn.disabled = true;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function showSection(section) {
        const sections = {
            upload: uploadSection,
            progress: progressSection,
            results: resultsSection,
            error: errorSection
        };

        Object.entries(sections).forEach(([key, el]) => {
            if (key === section) {
                el.style.display = 'block';
                el.style.animation = 'slide-up 0.4s ease';
            } else {
                el.style.display = 'none';
            }
        });

        // Scroll to the active section
        setTimeout(() => {
            sections[section].scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    function updateProgress(percent, message) {
        progressFill.style.width = percent + '%';
        progressPercent.textContent = percent + '%';
        progressMessage.textContent = message;
    }

    function showResults(results) {
        showSection('results');

        // Build stats cards
        const stats = [
            { value: results.clicks_detected.toLocaleString(), label: 'Clicks Found' },
            { value: results.clicks_per_second, label: 'Clicks / Sec' },
            { value: results.duration + 's', label: 'Duration' },
            { value: (results.sample_rate / 1000).toFixed(1) + 'k', label: 'Sample Rate' },
            { value: results.ar_order, label: 'AR Order' },
            { value: results.eta, label: 'Threshold η' },
        ];

        statsGrid.innerHTML = stats.map((s, i) => `
            <div class="stat-card" style="animation: slide-up 0.3s ease ${i * 0.05}s both;">
                <span class="stat-value">${s.value}</span>
                <span class="stat-label">${s.label}</span>
            </div>
        `).join('');

        // Audio players
        if (selectedFile) {
            const originalUrl = URL.createObjectURL(selectedFile);
            document.getElementById('original-audio').src = originalUrl;
        }
        document.getElementById('cleaned-audio').src = `/download/${currentJobId}`;

        // Download button
        downloadBtn.href = `/download/${currentJobId}`;

        // Plots
        if (results.plots) {
            for (const [key, base64Data] of Object.entries(results.plots)) {
                const img = document.getElementById(`plot-img-${key}`);
                if (img) {
                    img.src = `data:image/png;base64,${base64Data}`;
                }
            }
        }
    }

    function showError(message) {
        showSection('error');
        document.getElementById('error-message').textContent = message;
    }

    function resetApp() {
        currentJobId = null;
        clearFile();
        processBtn.disabled = true;
        updateProgress(0, 'Starting...');
        showSection('upload');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});