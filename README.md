# VL-JEPA: Self-Learning AGI Prototype (Final Fusion Edition)

**A biologically-inspired AI agent that learns, sleeps, dreams, and corrects its own logic.**

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

## 🛠️ Installation

1. **Clone the repository** (or save the script).
2. **Install Dependencies**:
   ```bash
   pip install torch numpy pillow
Run the Agent:

Bash
python3 vl_jepa_final_fusion.py
🎮 Interactive Commands
Once the agent finishes its "Infancy" phase (Physics Calibration), it enters Interactive Mode. You can type the following commands:

Command	Description	Example
explain: <phrase>	Introspects and describes a mental image. Triggers JIT learning if concepts are unknown.	
explain: large purple star

compare: <A> to <B>	Measures semantic and visual distance between two concepts.	
compare: red square to blue circle

drill: <A> vs <B>	Starts a contrastive training session to separate confusing concepts.	drill: triangle vs diamond

solidify	Sleep Mode. Runs a deep memory consolidation cycle to prevent forgetting.	solidify

auto	Agent autonomously picks unknown concepts and learns them.	auto

exam: <concept>	Generates a report card (25 tests) to verify mastery.	exam: red

show <z> <c> <s>	Debugs the visual cortex with a specific input.	show medium black circle

idle	Retries failed concepts from the queue.	idle

🧪 The "Turing Test" Script
To verify all AGI features (Perception, Agency, Memory, and Logic), run this sequence:

Plaintext
show medium red circle
exam: red
explain: large purple hexagon
learn: star
drill: triangle vs diamond
solidify
exam: red
compare: red square to blue circle
auto
Expected Behavior:

Physics: show calculates correct geometry (0 sides for circle).

Agency: explain triggers auto-learning for "Purple" and "Hexagon".

Plasticity: drill increases the distance between Triangle/Diamond.

Retention: solidify ensures exam: red remains high (>20/25) even after learning new shapes.

📂 Project Structure
vl_jepa_final_fusion.py: The complete, single-file source code.

final_fusion_logs/: Directory containing generated images and exam report cards.

🤖 How it Works (Under the Hood)
Visual Cortex: A 5-layer Hierarchical CNN processes 200x200 pixel images.

Latent Space: Concepts are mapped to orthogonal vectors to prevent "Mode Collapse."

Replay Buffer: Stores up to 10,000 past experiences.

Interleaved Training: Every training step mixes new data with old memories to maintain stability.

"The goal is not just to classify, but to understand."
