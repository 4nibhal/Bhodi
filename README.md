# Bhodi The IA powered doc explorer
Bhodi is a chatbot framework to work with vectorized documentation.

<p align="center">
  <img src="static_readme/bhodi.webp" alt="Bhodi">
</p>

---
## Set env
```bash
uv venv -p 3.10
source .venv/bin/activate
```
```bash
mkdir models # store your models here
mkdir data_to_vector # store the data to vectorize here
```
<div style="border-left: 4px solid #cfd6dd; padding: 8px 12px; margin: 10px 0;">
  <strong>⚠️ IMPORTANT!</strong>
  <p style="margin: 5px 0 0;">
    Check <a href="src/bhodi_doc_analyzer/config.py" style="color:#0056b3; text-decoration:none;">config.py</a> and configure your models, embedders, and tokenizers.
    I suggest putting your models in the <code>models</code> directory.
  </p>
  <p style="margin: 5px 0 0; font-size: 0.9em; color: #555;">
    Hint: You can download models from <a href="https://huggingface.co/models" style="color:#0056b3; text-decoration:none;">Huggingface</a>.
  </p>
</div>

## Install Packages  

### Packages
- Install pyproject packages
```bash
uv sync
```

### Llama.cpp
- For Nvidia CUDA:
```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
uv pip install llama-cpp-python
```
<div style="border: 1px solid #cfd6dd; padding: 8px 12px; margin: 10px 0;">
  <p style="margin: 5px 0;">
    <strong>Llama.cpp Installation:</strong>
    Check <a href="https://github.com/abetlen/llama-cpp-python" style="color:#0056b3; text-decoration:none;" target="_blank">official repo of llama.cpp</a>
    to see how to install lama.cpp for your hardware.
  </p>
</div>


### Build bhodi
- build our cli tool to index data in vectorstore and your caller for the chatbot
- execute this in root project directory
```bash
uv build
uv pip install dis dist/bhodi_doc_analyzer-0.1.0-py3-none-any.whl
```
- to use bhodi-index:
```bash
bhodi-index path/to/your/data # Support file or directory (only trully tested with PDF)
```

- to use bhodi as a chat bot
```bash
bhodi
```
---

---
## FEATURES

<details style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 1em;">
  <summary style="font-size: 1.1em; font-weight: bold; color:#cfd6dd;">🌟 ACTUAL FEATURES / IMPLEMENTED</summary>
  <ul style="margin-left: 20px;">
    <li>Basic chatbot TUI</li>
    <li>Vectorize/index several type of files</li>
    <li>Chat logs</li>
    <li>Easy to use</li>
    <li>Chat Memory with RAG</li>
  </ul>
</details>

<details style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 1em;">
  <summary style="font-size: 1.1em; font-weight: bold; color:#cfd6dd;">⚙️ NEEDED FEATURES / NEED TO BE IMPLEMENTED (PRIORITY)</summary>
  <ul style="margin-left: 20px;">
    <li>Copy MD blocks of code generated by the chat (maybe a widget in textual)</li>
    <li>Better indexing of files = more consistent and accurate RAG</li>
  </ul>
</details>

<details style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 1em;">
  <summary style="font-size: 1.1em; font-weight: bold; color:#cfd6dd;">💡 COULD BE USEFUL / NEED TO BE IMPLEMENTED (NOT PRIORITY)</summary>
  <ul style="margin-left: 20px;">
    <li>Embeding models using API, like google-gemini or OPENAI</li>
  </ul>
</details>