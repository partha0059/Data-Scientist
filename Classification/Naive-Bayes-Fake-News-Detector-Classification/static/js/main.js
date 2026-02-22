/**
 * Fake News Detector - Main JavaScript
 * Handles prediction requests and UI interactions
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const newsInput = document.getElementById('news-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const charCount = document.getElementById('char-count');
    const resultsContainer = document.getElementById('results-container');
    const resultCard = document.getElementById('result-card');
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultDescription = document.getElementById('result-description');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');
    const realProbValue = document.getElementById('real-prob-value');
    const fakeProbValue = document.getElementById('fake-prob-value');
    const btnText = document.querySelector('.btn-text');
    const btnIcon = document.querySelector('.btn-icon');
    const btnLoader = document.querySelector('.btn-loader');

    // Character counter
    newsInput.addEventListener('input', function() {
        charCount.textContent = this.value.length;
    });

    // Clear button
    clearBtn.addEventListener('click', function() {
        newsInput.value = '';
        charCount.textContent = '0';
        resultsContainer.classList.add('hidden');
        newsInput.focus();
    });

    // Analyze button click handler
    analyzeBtn.addEventListener('click', async function() {
        const newsText = newsInput.value.trim();

        // Validation
        if (!newsText) {
            showError('Please enter some news text to analyze.');
            newsInput.focus();
            return;
        }

        if (newsText.length < 20) {
            showError('Please enter at least 20 characters for accurate analysis.');
            return;
        }

        // Show loading state
        setLoading(true);

        try {
            // Make API request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: newsText })
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'An error occurred during analysis.');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Failed to connect to the server. Please try again.');
        } finally {
            setLoading(false);
        }
    });

    // Display results
    function displayResults(data) {
        resultsContainer.classList.remove('hidden');
        
        // Scroll to results
        setTimeout(() => {
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);

        // Update result card
        if (data.is_fake) {
            resultCard.className = 'result-card fake';
            resultIcon.textContent = '✕';
            resultTitle.textContent = 'FAKE NEWS DETECTED';
            resultTitle.style.animation = 'pulse 0.5s ease';
            resultDescription.textContent = 'This article shows characteristics commonly found in misinformation. Please verify from trusted sources.';
        } else {
            resultCard.className = 'result-card real';
            resultIcon.textContent = '✓';
            resultTitle.textContent = 'AUTHENTIC NEWS';
            resultTitle.style.animation = 'pulse 0.5s ease';
            resultDescription.textContent = 'This article appears to be authentic and follows patterns of legitimate news sources.';
        }

        // Animate confidence bar
        setTimeout(() => {
            confidenceValue.textContent = data.confidence + '%';
            confidenceFill.style.width = data.confidence + '%';
            
            // Color the confidence bar based on result
            if (data.is_fake) {
                confidenceFill.style.background = 'linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)';
            } else {
                confidenceFill.style.background = 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)';
            }
        }, 200);

        // Update probabilities with animation
        animateValue(realProbValue, 0, data.real_probability, 800);
        animateValue(fakeProbValue, 0, data.fake_probability, 800);
    }

    // Animate number value
    function animateValue(element, start, end, duration) {
        const startTime = performance.now();
        
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const current = start + (end - start) * easeOutQuart;
            
            element.textContent = current.toFixed(1) + '%';
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        
        requestAnimationFrame(update);
    }

    // Show error message
    function showError(message) {
        resultsContainer.classList.remove('hidden');
        resultCard.className = 'result-card fake';
        resultIcon.textContent = '⚠';
        resultTitle.textContent = 'Error';
        resultDescription.textContent = message;
        
        // Hide confidence and probability sections
        document.querySelector('.confidence-container').style.display = 'none';
        document.querySelector('.probability-container').style.display = 'none';
        
        setTimeout(() => {
            document.querySelector('.confidence-container').style.display = 'block';
            document.querySelector('.probability-container').style.display = 'flex';
        }, 3000);
    }

    // Set loading state
    function setLoading(isLoading) {
        analyzeBtn.disabled = isLoading;
        
        if (isLoading) {
            btnText.classList.add('hidden');
            btnIcon.classList.add('hidden');
            btnLoader.classList.remove('hidden');
            analyzeBtn.style.opacity = '0.8';
        } else {
            btnText.classList.remove('hidden');
            btnIcon.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            analyzeBtn.style.opacity = '1';
        }
    }

    // Smooth scroll for navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Update active nav link on scroll
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('section');
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (scrollY >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });

    // Add keyboard shortcut (Ctrl/Cmd + Enter to analyze)
    newsInput.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            analyzeBtn.click();
        }
    });

    // Add pulse animation keyframes dynamically
    const style = document.createElement('style');
    style.textContent = `
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
    `;
    document.head.appendChild(style);

    console.log('🔍 Fake News Detector initialized successfully!');
});
