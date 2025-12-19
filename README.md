## 🎮 Multi-Agent Reinforcement Learning in Real-Time Strategy (RTS)

**Coordinated Resource Collection via Q-Learning**

### 📌 Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system inspired by **Real-Time Strategy (RTS)** games. Multiple agents operate in a shared grid-based environment and **learn to collaboratively collect resources and deposit them at a base** using **tabular Q-learning** with ε-greedy exploration.

Despite having **no explicit communication**, agents exhibit **emergent cooperative behavior** through **reward shaping**, demonstrating how coordination can arise in decentralized systems.

The system is fully interactive, featuring:
* Real-time environment visualization
* Episode-wise and cumulative performance analytics
* Random-policy baseline comparison
* Agent-level and Q-table analysis

### ✨ Key Features
* 🧠 **Multi-Agent Q-Learning (Independent Q-Learning)**
* 🎯 **Reward shaping for coordination**
* 📉 **ε-greedy exploration with decay**
* 📊 **Training analytics & performance metrics**
* 🎮 **Live RTS-style grid visualization (Streamlit + Plotly)**
* 🎲 **Random policy baseline for comparison**
* 🚀 T**rain 1 Episode / Train 10 Episodes / Auto-Train modes**

### 🏗️ Environment Description
- **Grid-based RTS environment** (default: 15×15)
- **Multiple agents** (2–5 configurable)
- **Resource nodes** with finite quantities
- **Ally base** (collection target)
- **Enemy base** (for visualization)
- Episodes end when:
  * Maximum steps are reached (100), or
  * Required resources are deposited

The environment is **deterministic and Gym-like**, making it ideal for tabular RL experimentation.

### 🧠 Learning Algorithm

#### Multi-Agent Q-Learning (Independent Q-Learning)

Each agent independently maintains a Q-table and learns optimal state–action values through experience.

#### Q-Learning Update Rule
```bash
Q(s,a) ← Q(s,a) + α [r + γ · maxₐ′ Q(s′,a′) − Q(s,a)]
```
Where:
- α – Learning rate
- γ – Discount factor
- r – Immediate reward
- s′ – Next state

### 🎮 State & Action Space

#### State Representation


Each agent’s state is encoded as:
```bash
(x, y, carrying, nearest_resource_x, nearest_resource_y, base_x, base_y)
```
This compact representation keeps the state space manageable for **tabular learning**.

#### Action Space

Agents choose from:
- up, down, left, right
- collect
- deposit

### 🎁 Reward Structure
| Event              | Reward              | Purpose                     |
| ------------------ | ------------------- | --------------------------- |
| Collect resource   | +5                  | Encourage gathering         |
| Deposit at base    | +10                 | Reward successful delivery  |
| Coordination bonus | +2 per nearby agent | Encourage teamwork          |
| Movement penalty   | -0.1                | Discourage random wandering |

➡️ **Coordination bonus** is awarded when agents deposit resources while teammates are near the base.

### 🔍 Exploration Strategy
- **ε-greedy policy**
- High exploration initially
- Gradual decay:
```bash
ε ← max(ε_min, ε × decay)
```
- Smooth transition from exploration → exploitation

### 📊 Metrics & Analytics

Tracked metrics include:
- Episode reward
- Resources collected
- Success rate
- Cumulative reward
- Per-agent rewards
- Q-table size & statistics
- Action preference distribution
- Comparison with random-policy baseline

### 🎲 Random Policy Baseline

A **10-episode random-policy evaluation** is included to validate learning effectiveness.

| Metric        | Random Policy | Learned Policy       |
| ------------- | ------------- | -------------------- |
| Avg Reward    | Very low      | Significantly higher |
| Avg Resources | 1–3           | 10–15                |
| Success Rate  | ~0%           | 70–100%              |

### 🖥️ Visualization

The Streamlit UI provides:
- 🟩 Live grid visualization
- 🤖 Agent positions & status
- 💎 Resource nodes with remaining quantity
- 📈 Training curves & analytics dashboards
- 📊 Q-value histograms & action distributions

### ⚙️ Technologies Used
- **Python**
- **Streamlit** – UI & interaction
- **Plotly** – Visualization
- **NumPy / Pandas** – Computation & analytics
- **Collections (defaultdict)** – Q-table storage

### 🚀 How to Run
1️⃣ Install Dependencies
```bash 
pip install streamlit numpy pandas plotly
```

2️⃣ Run the Application
```bash 
streamlit run Multi_Agent_RTS.py
```

3️⃣ Use the UI
- Click **Initialize**
- Train using:
  - Train 1 Episode
  - Train 10 Episodes
  - Auto Train
- Compare with **Random Baseline**
- Analyze metrics and Q-table behavior

### 📌 Limitations
- Tabular Q-learning does not scale well to large environments
- No collision or obstacle handling
- Coordination depends on reward shaping
- No explicit communication between agents
- Environment is simplified compared to full RTS games

### 🔮 Future Work
- Deep RL (DQN, Actor-Critic)
- Centralized Training with Decentralized Execution (CTDE)
- Inter-agent communication
- Enemy agents (cooperative–competitive setting)
- Obstacles, fog-of-war, terrain
- Curriculum & transfer learning
