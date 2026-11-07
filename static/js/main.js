
document.addEventListener('DOMContentLoaded', () => {
  const promptInput = document.getElementById('promptInput');
  const charCounter = document.getElementById('charCounter');
  const clearBtn = document.getElementById('clearBtn');
  const generateBtn = document.getElementById('generateBtn');

  const loadingSection = document.getElementById('loadingSection');
  const loadingStageText = document.getElementById('loadingStageText');
  const elapsedTimeSpan = document.getElementById('elapsedTime');

  const errorSection = document.getElementById('errorSection');
  const errorType = document.getElementById('errorType');
  const errorMessage = document.getElementById('errorMessage');
  const errorDetails = document.getElementById('errorDetails');

  const resultSection = document.getElementById('resultSection');
  const generatedImage = document.getElementById('generatedImage');
  const downloadBtn = document.getElementById('downloadBtn');
  const submittedPromptText = document.getElementById('submittedPromptText');
  const copyPromptBtn = document.getElementById('copyPromptBtn');

  const metaModel = document.getElementById('metaModel');
  const metaDimensions = document.getElementById('metaDimensions');
  const metaDuration = document.getElementById('metaDuration');
  const metaBackend = document.getElementById('metaBackend');

  const chipButtons = document.querySelectorAll('.chip-btn');

  const diffusionStages = [
    'Preparing your image preview...',
    'Matching your prompt with available demo images...',
    'Loading demo preview...',
    'Finalizing your image preview...'
  ];


const demoImages = [
  {
    keywords: ['leopard', 'animal', 'tiger'],
    image: 'https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=800'
  },
  {
    keywords: ['mountain', 'nature', 'landscape'],
    image: 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800'
  },
  {
    keywords: ['city', 'building', 'architecture', 'future'],
    image: 'https://images.unsplash.com/photo-1519608487953-e999c86e7455?w=800'
  },
  {
    keywords: ['space', 'planet', 'galaxy', 'universe'],
    image: 'https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800'
  },
  {
    keywords: ['flower', 'garden'],
    image: 'https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=800'
  },
  {
    keywords: ['anime', 'cartoon', 'manga', 'animated', 'animation'],
    image: 'https://images.unsplash.com/photo-1578632767115-351597cf2477?w=800'
  },
  {
    keywords: ['human', 'person', 'people', 'man', 'woman', 'boy', 'girl', 'portrait', 'face'],
    image: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800'
  }
];

function findDemoImage(prompt) {
  const text = prompt.toLowerCase().trim();

  console.log("User prompt:", text);

  const matchedImage = demoImages.find(item =>
    item.keywords.some(keyword =>
      text.includes(keyword.toLowerCase())
    )
  );

  console.log("Matched image:", matchedImage);

  return matchedImage ? matchedImage : demoImages[1];
}

function updateCharCount() {
    const currentLength = promptInput.value.length;
    charCounter.textContent = `${currentLength} / 500`;
    charCounter.style.color =
      currentLength >= 480 ? 'var(--accent-rose)' : 'var(--text-muted)';
  }

  promptInput.addEventListener('input', updateCharCount);

  clearBtn.addEventListener('click', () => {
    promptInput.value = '';
    updateCharCount();
    promptInput.focus();
  });

  chipButtons.forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        promptInput.value = prompt;
        updateCharCount();
        promptInput.focus();
      }
    });
  });

  promptInput.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      handleGenerate();
    }
  });

  generateBtn.addEventListener('click', handleGenerate);

  if (copyPromptBtn) {
    copyPromptBtn.addEventListener('click', async () => {
      const textToCopy = submittedPromptText.textContent;
      if (!textToCopy) return;

      try {
        await navigator.clipboard.writeText(textToCopy);
        copyPromptBtn.textContent = 'Copied!';
      } catch (error) {
        console.error('Clipboard copy failed:', error);
      }
    });
  }


async function handleGenerate() {
  const rawPrompt = promptInput.value.trim();

  hideError();
  hideResult();

  if (!rawPrompt) {
    showError(
      'EmptyPromptError',
      'Prompt cannot be empty.',
      'Please enter an image description.'
    );
    promptInput.focus();
    return;
  }

  if (rawPrompt.length > 500) {
    showError(
      'PromptLengthExceededError',
      'Prompt exceeds 500 characters.',
      'Please shorten your prompt.'
    );
    return;
  }

  setLoadingState(true);

  try {
    console.log("Sending prompt to AI:", rawPrompt);

    const response = await fetch('/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt: rawPrompt
      })
    });

    const data = await response.json();

    if (!response.ok || data.status !== 'success') {
      throw new Error(
        data.message || 'Image generation failed.'
      );
    }

    console.log("AI generation successful:", data);

    showResult(data);

  } catch (error) {
    console.error("AI generation error:", error);

    showError(
      'ImageGenerationError',
      error.message || 'Unable to generate image.',
      'Check your Hugging Face API configuration and server logs.'
    );

  } finally {
    setLoadingState(false);
  }
}


  function showResult(data) {
    generatedImage.src = data.image_url;
    downloadBtn.href = data.image_url;
    downloadBtn.removeAttribute('download');
    submittedPromptText.textContent = data.prompt;

    metaModel.textContent = data.model;
    metaDimensions.textContent = data.dimensions;
    metaDuration.textContent = data.generation_time_seconds;
    metaBackend.textContent = data.execution_backend;

    resultSection.style.display = 'flex';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function showError(type, message, details) {
    errorType.textContent = type;
    errorMessage.textContent = message;
    errorDetails.textContent = details || '';
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideError() {
    errorSection.style.display = 'none';
  }

  function hideResult() {
    resultSection.style.display = 'none';
    generatedImage.src = '';
  }

  let timerInterval = null;
  let stageInterval = null;

  function setLoadingState(isLoading) {
    generateBtn.disabled = isLoading;
    clearBtn.disabled = isLoading;
    promptInput.disabled = isLoading;
    loadingSection.style.display = isLoading ? 'flex' : 'none';

    if (isLoading) {
      const startTime = Date.now();
      let stageIndex = 0;

      elapsedTimeSpan.textContent = '0.0s';
      loadingStageText.textContent = diffusionStages[0];

      timerInterval = setInterval(() => {
        elapsedTimeSpan.textContent =
          `${((Date.now() - startTime) / 1000).toFixed(1)}s`;
      }, 100);

      stageInterval = setInterval(() => {
        stageIndex = (stageIndex + 1) % diffusionStages.length;
        loadingStageText.textContent = diffusionStages[stageIndex];
      }, 3000);
    } else {
      clearInterval(timerInterval);
      clearInterval(stageInterval);
      timerInterval = null;
      stageInterval = null;
    }
  }

  updateCharCount();
});
