# VL-JEPA: Self-Learning AGI Prototype (Final Fusion Edition)

**A biologically-inspired AI agent that learns, sleeps, dreams, and corrects its own logic.**

![Alt text](kid-learning-shapes.jpg)

This project implements a **Vision-Language Joint Embedding Predictive Architecture (VL-JEPA)** that simulates early-stage cognitive development. Unlike standard classifiers, this agent possesses "Agency": it realizes when it doesn't know a concept, pauses to learn it, and uses a dual-process brain (Intuition vs. Logic) to prevent hallucinations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Status](https://img.shields.io/badge/Status-Stable-green)

## 🧠 Core Capabilities

### 1. Dual-Process Cognition (System 1 vs. System 2)
The agent compares its neural intuition against a hard-coded physics engine.
- **Intuition:** "I think I see a Triangle."
- **Logic:** "I count 4 sides."
- **Result:** It triggers a **Self-Correction Loop**, retraining itself until intuition matches reality.

### 2. Just-In-Time (JIT) Learning
If you ask `explain: large purple hexagon` and the agent doesn't know "purple," it won't crash or hallucinate. It will:
1.  Pause execution.
2.  Auto-generate a curriculum for "purple".
3.  Master the concept.
4.  Resume and answer your question.

### 3. Biological Memory & Sleep (`solidify`)
To solve **Catastrophic Forgetting** (e.g., learning "Blue" erases "Red"), the agent features a **Hippocampal Replay Buffer**. Running the `solidify` command puts the agent into "Deep Sleep," where it trains exclusively on past memories to consolidate long-term retention.

### 4. Semantic Drilling (`drill`)
When the agent confuses similar concepts (e.g., Triangle vs. Diamond), the `drill` mode forces a high-intensity contrastive training session until the semantic distance between the two concepts exceeds a safety threshold.

---
## 📊 Comparison: VL-JEPA vs. Other AI Architectures

| Feature | **This VL-JEPA (Final Fusion)** | **Standard I-JEPA (Meta)** | **LLMs (GPT, Gemini)** | **Traditional ML (ResNet)** |
| :--- | :--- | :--- | :--- | :--- |
| **Core Objective** | **Active Agency** (Realize ignorance & fix it) | **Representation** (Learn features from masking) | **Generation** (Predict the next word) | **Classification** (Map input to label) |
| **Cognitive Style** | **Dual-Process** (Intuition + Hard Logic) | Single-Process (Neural weights only) | Single-Process (Neural weights only) | Single-Process (Statistical mapping) |
| **Learning Method** | **Interleaved Replay** (Mixes new & old) | Self-Supervised (Masked Autoencoding) | Self-Supervised (Trillions of tokens) | Supervised (Human labels required) |
| **Data Efficiency** | **Extreme** (Learns from ~16 examples) | High | Low (Needs massive datasets) | Medium (Needs thousands of examples) |
| **Memory** | **Active Hippocampus** (Sleep/Solidify cycles) | Static Weights (Catastrophic forgetting) | Context Window (Short-term only) | Static Weights (Retrain to update) |
| **Hallucination** | **Self-Correcting** (Logic rejects wrong intuition) | N/A (Predicts embeddings, not pixels) | **Common** (Confidently wrong) | N/A (Just misclassifies) |
| **Plasticity** | **High** (Learns new concepts live) | Low (Pre-trained & Frozen) | None (Frozen after training) | None (Frozen after training) |
| **Compute Cost** | **Tiny** (Runs on CPU/Mac Mini) | High (Requires GPU clusters) | Massive (Data Centers) | Moderate (GPU helpful) |

---

### 🔑 Key Differences Explained

#### 1. "System 2" Logic (The Self-Correction Loop)
* **Most AI (LLMs/CNNs):** Rely 100% on "System 1" (Intuition). If an LLM thinks $2+2=5$, it just says it. It has no internal "calculator" to check its work before speaking.
* **This VL-JEPA:** Has a distinct "System 2" (The Physics Engine). When Intuition says "I see a Triangle," System 2 measures the angles. If they don't match, it **rejects its own thought** (`🚨 REJECTION!`) and retrains itself immediately.

#### 2. Agency & JIT Learning
* **Traditional ML:** If you show a ResNet a "Star" when it was only trained on "Circles," it will confidently classify it as a Circle. It has no concept of "Unknown."
* **This VL-JEPA:** Recognizes **Semantic Gaps**. If you ask about a "Star" and it has no embedding for it, it pauses, triggers a learning loop (`🚧 Learning 'star'...`), and creates the concept on the fly.

#### 3. Catastrophic Forgetting vs. Sleep
* **Standard Neural Nets:** If you train a model on "Task A" and then "Task B," it usually deletes "Task A" to make room.
* **This VL-JEPA:** Uses **Interleaved Batching** and **Sleep Cycles (`solidify`)**. It mimics the biological hippocampus by replaying old memories (Circles/Reds) while learning new ones (Stars/Purples), ensuring stability.

#### 4. Semantic Drilling
* **LLMs:** Often confuse similar concepts (e.g., specific coding syntax) and require fine-tuning to fix.
* **This VL-JEPA:** Can be ordered to **`drill`**. It enters a contrastive arena where it faces two confusing concepts (Triangle vs. Diamond) and is forced to mathematically separate them in its latent space until they are distinct.


---

## 🛠️ Installation

1. **Clone the repository** (or save the script).
2. **Install Dependencies**:
   ```bash
   pip install torch numpy pillow
   ```
3. **Run the Agent**:
   ```bash
   python3 vl_jepa_final_fusion.py
   ```

---

## 🎮 Interactive Commands

Once the agent finishes its "Infancy" phase (Physics Calibration), it enters **Interactive Mode**. You can type the following commands:

| Command | Description | Example |
| :--- | :--- | :--- |
| **`explain: <phrase>`** | Introspects and describes a mental image. Triggers JIT learning if concepts are unknown. | `explain: large purple star` |
| **`compare: <A> to <B>`** | Measures semantic and visual distance between two concepts. | `compare: red square to blue circle` |
| **`drill: <A> vs <B>`** | Starts a contrastive training session to separate confusing concepts. | `drill: triangle vs diamond` |
| **`solidify`** | **Sleep Mode.** Runs a deep memory consolidation cycle to prevent forgetting. | `solidify` |
| **`auto`** | Agent autonomously picks unknown concepts and learns them. | `auto` |
| **`exam: <concept>`** | Generates a report card (25 tests) to verify mastery. | `exam: red` |
| **`show <z> <c> <s>`** | Debugs the visual cortex with a specific input. | `show medium black circle` |
| **`idle`** | Retries failed concepts from the queue. | `idle` |

---

## 🧪 The "Turing Test" Script

To verify all AGI features (Perception, Agency, Memory, and Logic), run this sequence:

```text
show medium red circle
exam: red
explain: large purple hexagon
learn: star
drill: triangle vs diamond
solidify
exam: red
compare: red square to blue circle
auto
```

**Expected Behavior:**
1.  **Physics:** `show` calculates correct geometry (0 sides for circle).
2.  **Agency:** `explain` triggers auto-learning for "Purple" and "Hexagon".
3.  **Plasticity:** `drill` increases the distance between Triangle/Diamond.
4.  **Retention:** `solidify` ensures `exam: red` remains high (>20/25) even after learning new shapes.

---

## 📂 Project Structure

- **`vl_jepa_final_fusion.py`**: The complete, single-file source code.
- **`final_fusion_logs/`**: Directory containing generated images and exam report cards.

## 🤖 How it Works (Under the Hood)

1.  **Visual Cortex:** A 5-layer Hierarchical CNN processes 200x200 pixel images.
2.  **Latent Space:** Concepts are mapped to orthogonal vectors to prevent "Mode Collapse."
3.  **Replay Buffer:** Stores up to 10,000 past experiences.
4.  **Interleaved Training:** Every training step mixes new data with old memories to maintain stability.

---

> *"The goal is not just to classify, but to understand."*

## 🚀 v4.0 Update: Bootcamp Edition & Autonomous Agent

This release represents a major architectural leap, transitioning the system from a manual training tool to a **semi-autonomous agent**. It features a rewritten memory management system, "Focus Attention" mechanics for overcoming ViT shape bias, and native optimization for Apple Silicon (MPS).

### ✨ Key Features

#### 1. Smart Auto-Pilot (`auto`)
The `auto` command now supports flexible constraints (time or count) and includes **Smart Interleaving**. The agent automatically monitors its own short-term memory buffer and triggers a "Power Nap" (`solidify`) every 3 concepts to prevent catastrophic forgetting.

**Usage:**
- `auto` : Learns 3 random concepts (Default).
- `auto 10` : Learns 10 random concepts.
- `auto 5m` : Runs autonomously for 5 minutes.
- `auto 1h` : Runs autonomously for 1 hour.

#### 2. Selective Attention Mechanism ("Focus Mode")
Vision Transformers typically exhibit a strong "Shape Bias," often struggling to separate colors in the latent space. This version introduces a dynamic loss function:
- **Shape Mode:** Standard loss.
- **Color Mode:** Applies a **15x penalty** to color errors.
This forces the gradient descent algorithm to prioritize pixel intensity over edge detection during color drills, effectively "teaching" the AI to see color.

#### 3. Automated Bootcamps
New macro commands run round-robin tournaments to reinforce concept separation without manual input.
- `bootcamp color`: Drills every color pair (e.g., Red vs Blue, Blue vs Green) to maximize vector distance.
- `bootcamp shape`: Drills geometric distinctions (e.g., Square vs Rectangle).

#### 4. Mac/MPS Hardware Stabilization
- **Memory Stride Fix:** Implements `.contiguous()` checks on all tensor permutations to resolve the `RuntimeError: view size is not compatible` crash common on Apple Neural Engines.
- **Global Scope Resolution:** Moves core cognitive functions (`learn_concept`, `explain`, `solidify`) to the global scope to prevent `NameError` during extended autonomous sessions.

### 📋 Command Reference

| Command | Example | Description |
| :--- | :--- | :--- |
| **Auto** | `auto 30m` | Starts self-directed learning for a specific duration (m/h) or count. |
| **Bootcamp** | `bootcamp color` | Initiates an intensive round-robin drill for a specific domain. |
| **Drill** | `drill: red vs blue` | Manually forces contrastive learning between two specific concepts. |
| **Solidify** | `solidify` | Triggers "Deep Sleep" (Hippocampal Replay) to consolidate long-term memory. |
| **Explain** | `explain: large red star` | Inference test. If the agent encounters an unknown word, it halts to learn it first (JIT Learning). |
| **Compare** | `compare: red square to blue circle` | Returns the Euclidean distance between two concepts in the agent's latent space. |

# 🧠 VL-JEPA: Curriculum Edition (Mac Silicon Optimized)

![Status](https://img.shields.io/badge/Status-Stable-success)
![Hardware](https://img.shields.io/badge/Hardware-Apple_Silicon_MPS-gray?logo=apple)
![Architecture](https://img.shields.io/badge/Architecture-Neuro_Symbolic_ViT-blue)

A biologically inspired Artificial General Intelligence (AGI) prototype designed to run natively on Apple Silicon (M1/M2/M3). 

Unlike standard neural networks, this agent possesses **metacognition** (it knows what it doesn't know), **hippocampal memory** (it sleeps to consolidate knowledge), and a **curriculum knowledge graph** (it learns simple concepts before complex ones).

## 🌟 Key Features

* **Mac/MPS Native:** Optimized memory strides (`.contiguous()`) to prevent `RuntimeError` crashes on the Apple Neural Engine.
* **Neuro-Symbolic Core:** Combines a Vision Transformer (Intuition/System 1) with a Physics Engine (Logic/System 2). Logic overrides hallucination.
* **Curriculum Learning:** Uses a directed acyclic graph (DAG) to map dependencies. The agent will not attempt to learn "Brown" until it has mastered "Red" and "Green."
* **Active Agency:** If asked to explain an unknown concept, the agent halts, triggers a Just-In-Time (JIT) learning loop, masters the concept, and *then* answers.
* **Selective Attention:** Overcomes the standard Vision Transformer "Shape Bias" by dynamically re-weighting the loss function during Color Bootcamps.
* **Robust Auto-Pilot:** Can run unsupervised for set counts or durations, managing its own sleep cycles to prevent Catastrophic Forgetting.

## 🛠️ Installation & Usage

### Prerequisites
You need Python 3 installed along with PyTorch and Pillow.

```bash
pip install torch numpy pillow
```
Running the Agent

Ensure you are running this on a Mac with Apple Silicon for optimal performance (though it will fallback to CPU if needed).

```bash
python3 vl_jepa_curriculum_final.py
```

### 🤖 Command Reference
Once the agent is online, interact with it using these natural language commands:

#### 1. Auto-Pilot (Unsupervised Learning)

The agent autonomously explores the universe, picking concepts based on its Knowledge Graph and managing memory.

auto: Learns 3 concepts (Default).

auto 10: Learns 10 concepts, sleeping every 3 items.

auto 30m: Runs continuously for 30 minutes.

auto 1h: Runs continuously for 1 hour.

#### 2. Agency & Curriculum

consult: The agent analyzes its own knowledge gaps and suggests the next logical step (e.g., "Master! I know Red and Blue. May I learn Purple?").

explain: [text]: Asks the agent to visualize and analyze a complex query (e.g., explain: small brown octagon). If it lacks knowledge, it will auto-learn the missing pieces before answering.

#### 3. Training Modules

bootcamp color: Forces a round-robin tournament of all color pairs with high attention penalties. Use this if the agent struggles to distinguish colors.

bootcamp shape: Rapid-fire contrastive learning for geometry.

drill: [A] vs [B]: Manually triggers a contrastive session between two concepts (e.g., drill: red vs blue).

learn: [concept]: Forces the agent to focus on a specific concept until mastery (25/25 score).

#### 4. Memory Management

solidify: Triggers "Deep Sleep." The agent replays randomized memories from its buffer to consolidate short-term weights into long-term stability. (Automatically triggered during auto runs).

#### 5. Diagnostics

exam: [concept]: Runs a 25-question test on a specific concept.

compare: [A] to [B]: Measures the semantic distance between two concepts in the agent's latent space.

show [size] [color] [shape]: Force-feeds an image to the "Eye" and probes System 1 (Intuition) vs System 2 (Logic).

### 🧬 Architecture: System 1 vs. System 2
The agent makes decisions using two distinct systems, modeled after human cognition:

Component	Architecture	Role	Behavior
System 1	Vision Transformer (ViT)	Intuition	Fast, pattern-matching, prone to hallucination. Sees "Red" and "Blue" as mathematical vectors.
System 2	Symbolic Physics Engine	Logic	Slow, rule-based, infallible. Measures geometry (sides, angles) to verify System 1.
The Interaction: If System 1 says "I see a Triangle" but System 2 measures 4 sides, the agent triggers a 🚨 REJECTION event, overrides the neural network, and prevents the hallucination from being reinforced.

### 🗺️ Extending the Universe
The agent lives in a "Closed World" defined by the physics dictionaries at the top of the script. To add new concepts, edit vl_jepa_curriculum_final.py:

#### 1. Add Geometry: Update SHAPE_SIDES:

```Python
"decagon": 10.0,
"line": 1.0
```

#### 2. Add Physics: Update draw_tensor to define how to render the new shape via PIL:

```Python
elif shape == "decagon":
    # logic to draw 10 sides...
```
#### 3. Add Dependencies: Update CURRICULUM to teach the agent the hierarchy:

```Python
CURRICULUM = {
    "decagon": ["pentagon"], # Decagon requires knowledge of Pentagon
    ...
}
```

