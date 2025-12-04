---

# **Emergency Triage Assistant**

AI-powered emergency-response triage system using **local LLMs, ML classification, RAG safety guidance**, and a fully interactive chat UI.

> This project simulates how a real emergency-dispatch assistant collects critical information, analyzes severity, identifies risks, and guides the user—fully offline using local models.

---

## **📌 Key Features**

### **🚨 Intelligent Crisis Detection**

* Classifies incidents into: **Flood, Fire, Earthquake, Storm, Landslide, Other**
* Detects **severity (low / medium / high)** using rule-based + LLM severity refinement
* Detects risk factors such as:

  * trapped / cannot exit
  * smoke inhalation
  * unconsciousness
  * heavy bleeding

### **🧠 Local LLM Pipeline (No Cloud Required)**

* Summarization, severity probing, and reply generation all run through **Ollama (llama3)**.
* Automatic fallback to deterministic logic when the LLM times out.

### **📍 Location Extraction**

Extracts street addresses and landmarks using regex + heuristics:

```
"225 Llama Street" → valid address
"near the station" → landmark
```

### **📚 RAG Safety Guidance**

A small Retrieval-Augmented-Generation system with `kb.json`:

* Category-specific guidance
* Generic fallback if nothing matches

### **💬 Chat UX**

* Full chat interface with typing indicator
* Auto-scroll
* Real-time right-pane “Incident Overview” with:

  * Category
  * Severity
  * Location
  * Guidance
  * Summary
  * People affected
  * Risk factors

### **🧪 Full Test Suite**

* API tests
* Classifier tests
* RAG tests
* Deterministic fallbacks validated

---

# **🖼️ System Architecture**

```
┌──────────────────────────┐
│        Frontend          │  static/index.html
│  - Chat UI               │
│  - Typing indicator      │
│  - Right-side dashboard  │
└──────────────┬───────────┘
               │ POST /triage (full conversation)
               ▼
┌──────────────────────────────┐
│          FastAPI API         │ app/api.py
│  - Validates request         │
│  - Calls TriageAgent         │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│         TriageAgent          │ app/agent.py
│  - Builds summary            │
│  - Passes text → pipeline    │
│  - Selects LLM reply         │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│       TriagePipeline         │ app/pipeline.py
│  - Crisis classification     │
│  - Severity adjustment       │
│  - LLM severity probe        │
│  - Location extraction       │
│  - RAG guidance retrieval    │
│  - Fallback reply builder    │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────┐
│  Models & Components     │
│  • TF-IDF classifier     │
│  • Optional transformer  │
│  • Location extractor    │
│  • RAG                   │
│  • LLM summarizer        │
│  • LLM severity checker  │
│  • LLM reply generator   │
└──────────────────────────┘
```

---

# **🚀 Installation**

### **1. Clone**

```bash
git clone https://github.com/AlirezaFarkhondeh2k3/emergency-triage-assistant
cd emergency-triage-assistant
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Install a local LLM (required for full functionality)**

```bash
ollama pull llama3
```

### **4. Run the server**

```bash
uvicorn app.api:app --reload
```

### **5. Open the UI**

```
http://127.0.0.1:8000
```

Your app will load with the chat interface and live dashboard.

---

# **⚙️ Detailed Pipeline Overview**

This is the **exact step-by-step flow** used in the backend.

### **1. User sends a message**

Frontend sends full message history to **POST /triage**.

### **2. TriageAgent reconstructs context**

* Extracts latest user message
* Summarizes conversation via:

  * llama3 (if available)
  * fallback: trimmed user text

### **3. Pipeline classification**

The text passes through:

#### **a. Crisis classification**

* TF-IDF baseline model
* Transformer classifier (optional via USE_TRANSFORMER)
* Keyword overrides for:

  * flood
  * smoke → fire
  * water keywords
  * basement keywords

#### **b. Severity inference**

* Rule-based if dangerous clues found
* LLM severity model (llama3)
* Escalates severity when:

  * trapped
  * unconscious
  * heavy bleeding
  * smoke inhalation
  * children involved in flood

#### **c. Location extraction**

Regex + NLP heuristics.

#### **d. RAG guidance**

Matching on:

* category + tags
* category only
* fallback

#### **e. Reply construction**

If LLM reply works → use it
Else → deterministic fallback paragraph.

### **4. Response returned to frontend**

Includes:

* reply
* category
* severity
* location
* guidance
* summary

### **5. UI updates**

Right-pane dashboard updates immediately.

---

# **🧪 Running Tests**

```
pytest -q
```

Covers:

* API contract
* Classification logic
* RAG fallback behavior

---

# **📁 Project Structure**

```
app/
  api.py
  agent.py
  pipeline.py
  config.py
  models/
      artifacts/
      transformer_artifacts/
      rag_artifacts/
      classifier.py
      location_extractor.py
      rag.py
      reply_llm.py
      severity_llm.py
      summarizer.py
static/
  index.html
tests/
  test_api.py
  test_classifier.py
  test_rag.py
requirements.txt
```

---

# **📺 Demo / Preview**

<img width="1089" height="1120" alt="image" src="https://github.com/user-attachments/assets/c31ef0ff-d831-4bf5-839a-a5a7440f5428" />

---

# **🤝 Contributing**

Pull requests are welcome.
Issues can be opened for bugs or feature requests.

---

# **📝 License**

MIT License.

---

# **✨ Author**

**Alireza Farkhondeh**
Machine Learning & AI Engineer

---

