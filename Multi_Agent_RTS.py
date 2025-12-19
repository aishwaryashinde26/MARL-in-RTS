import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import defaultdict
import random

# Page configuration
st.set_page_config(
    page_title="Multi-Agent RL in RTS",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.episode = 0
    st.session_state.step = 0
    st.session_state.auto_train = False
    st.session_state.q_table = defaultdict(
        lambda: {a: 0.0 for a in ['up', 'down', 'left', 'right', 'collect', 'deposit']}
    )
    st.session_state.training_history = []
    st.session_state.agents = []
    st.session_state.resources = []
    st.session_state.bases = []
    st.session_state.metrics = {
        'total_reward': 0,
        'resources_collected': 0,
        'success_rate': 0,
        'avg_coordination': 0
    }
    # Exploration schedule
    st.session_state.epsilon_current = 0.3
    st.session_state.epsilon_min = 0.05
    st.session_state.epsilon_decay = 0.99
    # Random baseline cache
    st.session_state.random_baseline = None

# -------------------------------------------------------------------
# Helper Classes
# -------------------------------------------------------------------
class Agent:
    def __init__(self, agent_id, x, y, color):
        self.id = agent_id
        self.x = x
        self.y = y
        self.carrying = False
        self.resource_amount = 0
        self.color = color
        self.total_reward = 0
        self.history = []


class Resource:
    def __init__(self, x, y, amount):
        self.x = x
        self.y = y
        self.amount = amount
        self.active = True


class Base:
    def __init__(self, x, y, team):
        self.x = x
        self.y = y
        self.team = team
        self.collected = 0


# -------------------------------------------------------------------
# RL Functions
# -------------------------------------------------------------------
def get_state_key(agent, resources, base, grid_size):
    """Generate state representation for Q-learning"""
    active_resources = [r for r in resources if r.active and r.amount > 0]

    if active_resources:
        nearest_resource = min(
            active_resources,
            key=lambda r: abs(agent.x - r.x) + abs(agent.y - r.y)
        )
        res_x, res_y = nearest_resource.x, nearest_resource.y
    else:
        res_x, res_y = -1, -1

    return f"{agent.x},{agent.y},{int(agent.carrying)},{res_x},{res_y},{base.x},{base.y}"


def select_action(agent, resources, base, q_table, epsilon, grid_size):
    """Epsilon-greedy action selection"""
    state_key = get_state_key(agent, resources, base, grid_size)
    actions = ['up', 'down', 'left', 'right', 'collect', 'deposit']

    if random.random() < epsilon:
        return random.choice(actions)
    else:
        q_values = q_table[state_key]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)


def execute_action(agent, action, resources, base, agents, grid_size):
    """Execute action and return reward"""
    reward = -0.1  # Small movement penalty
    new_x, new_y = agent.x, agent.y
    coord_bonus = 0

    if action == 'up':
        new_y = max(0, agent.y - 1)
    elif action == 'down':
        new_y = min(grid_size - 1, agent.y + 1)
    elif action == 'left':
        new_x = max(0, agent.x - 1)
    elif action == 'right':
        new_x = min(grid_size - 1, agent.x + 1)
    elif action == 'collect':
        for resource in resources:
            if (
                resource.x == agent.x and resource.y == agent.y
                and resource.active and resource.amount > 0
            ):
                if not agent.carrying:
                    agent.carrying = True
                    agent.resource_amount = 1
                    resource.amount -= 1
                    if resource.amount == 0:
                        resource.active = False
                    reward = 5.0
                break
    elif action == 'deposit':
        if base.x == agent.x and base.y == agent.y and agent.carrying:
            agent.carrying = False
            base.collected += agent.resource_amount
            reward = 10.0
            agent.resource_amount = 0

            # Coordination bonus: count nearby agents around base
            nearby_agents = sum(
                1
                for a in agents
                if a.id != agent.id
                and abs(a.x - base.x) < 3
                and abs(a.y - base.y) < 3
            )
            coord_bonus = nearby_agents * 2

    agent.x = new_x
    agent.y = new_y
    total_reward = reward + coord_bonus
    agent.total_reward += total_reward

    return total_reward


def update_q_value(q_table, state_key, action, reward, next_state_key, lr, gamma):
    """Q-learning update rule"""
    current_q = q_table[state_key][action]
    next_max_q = max(q_table[next_state_key].values())
    new_q = current_q + lr * (reward + gamma * next_max_q - current_q)
    q_table[state_key][action] = new_q


def initialize_environment(num_agents, grid_size):
    """Initialize agents, resources, and bases"""
    agents = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    for i in range(num_agents):
        agents.append(Agent(i, 1 + i, 1, colors[i % len(colors)]))

    resources = []
    for _ in range(8):
        resources.append(Resource(
            random.randint(2, grid_size - 4),
            random.randint(2, grid_size - 4),
            random.randint(2, 5)
        ))

    bases = [
        Base(1, 1, 'ally'),
        Base(grid_size - 2, grid_size - 2, 'enemy')
    ]

    return agents, resources, bases


def train_one_episode(
    agents,
    resources,
    bases,
    q_table,
    epsilon,
    learning_rate,
    gamma,
    grid_size,
    visualize=False,
    placeholder=None
):
    """Train for one complete episode using current policy"""
    episode_reward = 0

    for step in range(100):
        if bases[0].collected >= 8:
            break

        for agent in agents:
            state_key = get_state_key(agent, resources, bases[0], grid_size)
            action = select_action(agent, resources, bases[0], q_table, epsilon, grid_size)
            reward = execute_action(agent, action, resources, bases[0], agents, grid_size)
            next_state_key = get_state_key(agent, resources, bases[0], grid_size)
            update_q_value(q_table, state_key, action, reward, next_state_key, learning_rate, gamma)
            episode_reward += reward

        # Visualize every 5 steps if enabled
        if visualize and placeholder and step % 5 == 0:
            with placeholder.container():
                fig_env = create_environment_plot(agents, resources, bases, grid_size)
                st.plotly_chart(fig_env, use_container_width=True, key=f"env_plot_{step}")
                st.caption(
                    f"Step {step}/100 | Collected: {bases[0].collected} "
                    f"| Episode Reward: {episode_reward:.1f}"
                )
            time.sleep(0.1)

    success = 1 if bases[0].collected >= 8 else 0
    collected = bases[0].collected

    return episode_reward, collected, success


def run_random_episode(num_agents, grid_size, visualize=False, placeholder=None):
    """Run one episode with a purely random policy (baseline)."""
    agents, resources, bases = initialize_environment(num_agents, grid_size)
    episode_reward = 0

    for step in range(100):
        if bases[0].collected >= 8:
            break

        for agent in agents:
            action = random.choice(['up', 'down', 'left', 'right', 'collect', 'deposit'])
            reward = execute_action(agent, action, resources, bases[0], agents, grid_size)
            episode_reward += reward

        if visualize and placeholder and step % 5 == 0:
            with placeholder.container():
                fig_env = create_environment_plot(agents, resources, bases, grid_size)
                st.plotly_chart(fig_env, use_container_width=True)
                st.caption(
                    f"[Random] Step {step}/100 | Collected: {bases[0].collected} "
                    f"| Episode Reward: {episode_reward:.1f}"
                )
            time.sleep(0.1)

    collected = bases[0].collected
    return episode_reward, collected


# -------------------------------------------------------------------
# Visualization Helpers
# -------------------------------------------------------------------
def create_environment_plot(agents, resources, bases, grid_size):
    """Create interactive plotly visualization of the environment"""
    fig = go.Figure()

    # Grid
    for i in range(grid_size + 1):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[0, grid_size],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[0, grid_size], y=[i, i],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Bases
    for base in bases:
        color = '#10b981' if base.team == 'ally' else '#ef4444'
        fig.add_trace(go.Scatter(
            x=[base.x], y=[base.y],
            mode='markers+text',
            marker=dict(size=40, color=color, symbol='square'),
            text=[f'Base<br>{base.collected}'],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            name=f'{base.team.capitalize()} Base',
            hovertext=f'{base.team.capitalize()} Base<br>Collected: {base.collected}'
        ))

    # Resources
    active_resources = [r for r in resources if r.active and r.amount > 0]
    if active_resources:
        fig.add_trace(go.Scatter(
            x=[r.x for r in active_resources],
            y=[r.y for r in active_resources],
            mode='markers+text',
            marker=dict(size=25, color='#fbbf24', symbol='diamond'),
            text=[str(r.amount) for r in active_resources],
            textposition='top center',
            name='Resources',
            hovertext=[f'Resource<br>Amount: {r.amount}' for r in active_resources]
        ))

    # Agents
    if agents:
        fig.add_trace(go.Scatter(
            x=[a.x for a in agents],
            y=[a.y for a in agents],
            mode='markers+text',
            marker=dict(size=30, color=[a.color for a in agents], symbol='circle'),
            text=[f'A{a.id}' for a in agents],
            textposition='bottom center',
            name='Agents',
            hovertext=[
                f"Agent {a.id}<br>Pos: ({a.x},{a.y})<br>Carrying: {a.carrying}"
                for a in agents
            ]
        ))

    fig.update_layout(
        xaxis=dict(range=[0, grid_size], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, grid_size], showgrid=False, zeroline=False),
        width=600,
        height=600,
        showlegend=True,
        legend=dict(x=1.05, y=1, bgcolor='rgba(255,255,255,0.1)', font=dict(color='white')),
        font=dict(color='white'),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        hovermode='closest'
    )

    return fig


def create_training_plot(history):
    """Create training progress visualization"""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Resources Collected', 'Success Rate', 'Cumulative Rewards'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )

    # Episode Rewards
    fig.add_trace(
        go.Scatter(
            x=df['episode'], y=df['reward'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='#667eea', width=3)
        ),
        row=1, col=1
    )

    # Resources Collected
    fig.add_trace(
        go.Scatter(
            x=df['episode'], y=df['collected'],
            mode='lines+markers',
            name='Collected',
            line=dict(color='#fbbf24', width=3)
        ),
        row=1, col=2
    )

    # Success Rate (rolling average)
    if len(df) >= 5:
        df['success_rate'] = df['success'].rolling(window=5, min_periods=1).mean() * 100
        fig.add_trace(
            go.Scatter(
                x=df['episode'], y=df['success_rate'],
                mode='lines+markers',
                name='Success %',
                line=dict(color='#10b981', width=3)
            ),
            row=2, col=1
        )

    # Cumulative Rewards
    df['cumulative_reward'] = df['reward'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=df['episode'], y=df['cumulative_reward'],
            mode='lines',
            name='Cumulative',
            line=dict(color='#764ba2', width=3),
            fill='tozeroy'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='white')
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

    return fig


# -------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------
def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 Multi-Agent Reinforcement Learning in RTS</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Q-Learning with Coordinated Resource Collection</p>',
                unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration")

    num_agents = st.sidebar.slider("Number of Agents", 2, 5, 3)
    grid_size = st.sidebar.slider("Grid Size", 10, 20, 15)
    learning_rate = st.sidebar.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01)

    # Exploration schedule controls
    epsilon_start = st.sidebar.slider("Initial Exploration (ε₀)", 0.1, 1.0, 0.3, 0.05)
    epsilon_min = st.sidebar.slider("Minimum ε", 0.01, 0.5, 0.05, 0.01)
    epsilon_decay = st.sidebar.slider("ε Decay (per episode)", 0.90, 0.999, 0.99, 0.005)
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.8, 0.99, 0.95, 0.01)

    # Sync session epsilon parameters
    if 'epsilon_current' not in st.session_state:
        st.session_state.epsilon_current = epsilon_start
    st.session_state.epsilon_min = epsilon_min
    st.session_state.epsilon_decay = epsilon_decay

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Objective")
    st.sidebar.info(
        "Agents must learn to collect resources and deposit them at the base through coordinated behavior."
    )

    st.sidebar.markdown("### 🏆 Reward Structure")
    st.sidebar.markdown("""
    - **+5**: Collect resource  
    - **+10**: Deposit at base  
    - **+2/agent**: Coordination bonus  
    - **-0.1**: Movement penalty
    """)

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Current ε:** `{st.session_state.epsilon_current:.3f}`")

    # Control Buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🎬 Initialize", use_container_width=True):
            st.session_state.agents, st.session_state.resources, st.session_state.bases = \
                initialize_environment(num_agents, grid_size)
            st.session_state.step = 0
            st.session_state.metrics['total_reward'] = 0
            st.session_state.episode = 0
            st.session_state.training_history = []
            st.session_state.q_table = defaultdict(
                lambda: {a: 0.0 for a in ['up', 'down', 'left', 'right', 'collect', 'deposit']}
            )
            st.session_state.random_baseline = None
            st.session_state.epsilon_current = epsilon_start
            st.rerun()

    with col2:
        if st.button("▶️ Train 1 Episode", use_container_width=True):
            if not st.session_state.agents:
                st.session_state.agents, st.session_state.resources, st.session_state.bases = \
                    initialize_environment(num_agents, grid_size)

            viz_placeholder = st.empty()

            episode_reward, collected, success = train_one_episode(
                st.session_state.agents,
                st.session_state.resources,
                st.session_state.bases,
                st.session_state.q_table,
                st.session_state.epsilon_current,
                learning_rate,
                gamma,
                grid_size,
                visualize=True,
                placeholder=viz_placeholder
            )

            st.session_state.training_history.append({
                'episode': st.session_state.episode,
                'reward': episode_reward,
                'collected': collected,
                'success': success
            })

            st.session_state.episode += 1
            st.session_state.metrics['total_reward'] = episode_reward
            st.session_state.metrics['resources_collected'] = collected

            # Update success rate
            if st.session_state.training_history:
                successes = sum(h['success'] for h in st.session_state.training_history)
                st.session_state.metrics['success_rate'] = \
                    (successes / len(st.session_state.training_history)) * 100

            # Epsilon decay
            st.session_state.epsilon_current = max(
                st.session_state.epsilon_min,
                st.session_state.epsilon_current * st.session_state.epsilon_decay
            )

            viz_placeholder.empty()
            st.session_state.agents, st.session_state.resources, st.session_state.bases = \
                initialize_environment(num_agents, grid_size)

            st.success(
                f"✅ Episode {st.session_state.episode - 1} completed! "
                f"Reward: {episode_reward:.1f} | Collected: {collected}"
            )
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("⚡ Train 10 Episodes", use_container_width=True):
            if not st.session_state.agents:
                st.session_state.agents, st.session_state.resources, st.session_state.bases = \
                    initialize_environment(num_agents, grid_size)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(10):
                episode_reward, collected, success = train_one_episode(
                    st.session_state.agents,
                    st.session_state.resources,
                    st.session_state.bases,
                    st.session_state.q_table,
                    st.session_state.epsilon_current,
                    learning_rate,
                    gamma,
                    grid_size
                )

                st.session_state.training_history.append({
                    'episode': st.session_state.episode,
                    'reward': episode_reward,
                    'collected': collected,
                    'success': success
                })

                st.session_state.episode += 1
                st.session_state.metrics['total_reward'] = episode_reward
                st.session_state.metrics['resources_collected'] = collected

                # Epsilon decay after each episode
                st.session_state.epsilon_current = max(
                    st.session_state.epsilon_min,
                    st.session_state.epsilon_current * st.session_state.epsilon_decay
                )

                st.session_state.agents, st.session_state.resources, st.session_state.bases = \
                    initialize_environment(num_agents, grid_size)

                progress_bar.progress((i + 1) / 10)
                status_text.text(f"Training Episode {i + 1}/10")

            if st.session_state.training_history:
                successes = sum(h['success'] for h in st.session_state.training_history)
                st.session_state.metrics['success_rate'] = \
                    (successes / len(st.session_state.training_history)) * 100

            progress_bar.empty()
            status_text.empty()
            st.success("✅ Completed 10 training episodes!")
            time.sleep(1)
            st.rerun()

    with col4:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.episode = 0
            st.session_state.training_history = []
            st.session_state.q_table = defaultdict(
                lambda: {a: 0.0 for a in ['up', 'down', 'left', 'right', 'collect', 'deposit']}
            )
            st.session_state.metrics = {
                'total_reward': 0,
                'resources_collected': 0,
                'success_rate': 0,
                'avg_coordination': 0
            }
            st.session_state.agents, st.session_state.resources, st.session_state.bases = \
                initialize_environment(num_agents, grid_size)
            st.session_state.auto_train = False
            st.session_state.epsilon_current = epsilon_start
            st.session_state.random_baseline = None
            st.rerun()

    with col5:
        auto_train_label = "🚀 Auto Train" if not st.session_state.get('auto_train', False) else "⏸️ Stop Auto"
        if st.button(auto_train_label, use_container_width=True):
            st.session_state.auto_train = not st.session_state.get('auto_train', False)
            st.rerun()

    # Auto training loop
    if st.session_state.get('auto_train', False):
        if not st.session_state.agents:
            st.session_state.agents, st.session_state.resources, st.session_state.bases = \
                initialize_environment(num_agents, grid_size)

        st.info("🔄 Auto-training in progress. Click 'Stop Auto' to pause.")

        auto_viz_placeholder = st.empty()

        episode_reward, collected, success = train_one_episode(
            st.session_state.agents,
            st.session_state.resources,
            st.session_state.bases,
            st.session_state.q_table,
            st.session_state.epsilon_current,
            learning_rate,
            gamma,
            grid_size,
            visualize=True,
            placeholder=auto_viz_placeholder
        )

        st.session_state.training_history.append({
            'episode': st.session_state.episode,
            'reward': episode_reward,
            'collected': collected,
            'success': success
        })

        st.session_state.episode += 1
        st.session_state.metrics['total_reward'] = episode_reward
        st.session_state.metrics['resources_collected'] = collected

        if st.session_state.training_history:
            successes = sum(h['success'] for h in st.session_state.training_history)
            st.session_state.metrics['success_rate'] = \
                (successes / len(st.session_state.training_history)) * 100

        # Epsilon decay
        st.session_state.epsilon_current = max(
            st.session_state.epsilon_min,
            st.session_state.epsilon_current * st.session_state.epsilon_decay
        )

        auto_viz_placeholder.empty()
        st.session_state.agents, st.session_state.resources, st.session_state.bases = \
            initialize_environment(num_agents, grid_size)

        time.sleep(0.3)
        st.rerun()

    # Random baseline button (10 episodes)
    st.markdown("### 🎲 Baseline Comparison")
    if st.button("Run 10-Episode Random Policy Baseline"):
        rewards = []
        resources_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(10):
            r, c = run_random_episode(num_agents, grid_size, visualize=False)
            rewards.append(r)
            resources_list.append(c)
            progress_bar.progress((i + 1) / 10)
            status_text.text(f"Random Episode {i + 1}/10")

        progress_bar.empty()
        status_text.empty()
        st.session_state.random_baseline = {
            'avg_reward': float(np.mean(rewards)),
            'avg_resources': float(np.mean(resources_list))
        }
        st.success(
            f"✅ Random baseline complete. "
            f"Avg Reward: {np.mean(rewards):.2f}, Avg Resources: {np.mean(resources_list):.2f}"
        )

    # Metrics Dashboard
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Episode", st.session_state.episode)

    with col2:
        st.metric("Last Episode Reward", f"{st.session_state.metrics['total_reward']:.1f}")

    with col3:
        st.metric("Last Resources", st.session_state.metrics['resources_collected'])

    with col4:
        st.metric("Success Rate", f"{st.session_state.metrics['success_rate']:.1f}%")

    with col5:
        st.metric("Current ε", f"{st.session_state.epsilon_current:.3f}")

    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎮 Environment", "📈 Training Progress", "🤖 Agent Analysis", "📚 Documentation"]
    )

    # ----------------- Environment Tab -----------------
    with tab1:
        st.markdown("### Current Environment State")

        if st.session_state.agents:
            fig_env = create_environment_plot(
                st.session_state.agents,
                st.session_state.resources,
                st.session_state.bases,
                grid_size
            )
            st.plotly_chart(fig_env, use_container_width=True)

            st.markdown("### Agent Status")
            agent_data = []
            for agent in st.session_state.agents:
                agent_data.append({
                    'Agent ID': agent.id,
                    'Position': f'({agent.x}, {agent.y})',
                    'Carrying': '📦' if agent.carrying else '❌',
                    'Total Reward': f'{agent.total_reward:.2f}'
                })
            st.dataframe(pd.DataFrame(agent_data), use_container_width=True)
        else:
            st.warning("Click 'Initialize' to create the environment.")

    # ----------------- Training Progress Tab -----------------
    with tab2:
        st.markdown("### Training Progress Over Episodes")

        if st.session_state.training_history:
            fig_training = create_training_plot(st.session_state.training_history)
            st.plotly_chart(fig_training, use_container_width=True)

            df = pd.DataFrame(st.session_state.training_history)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Reward", f"{df['reward'].mean():.2f}")
            with col2:
                st.metric("Max Resources", int(df['collected'].max()))
            with col3:
                st.metric("Total Episodes", len(df))

            # Last-20 window stats for better analysis
            if len(df) >= 5:
                last_n = min(20, len(df))
                last_df = df.tail(last_n)
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric(f"Avg Reward (Last {last_n})", f"{last_df['reward'].mean():.2f}")
                with col5:
                    st.metric(
                        f"Avg Resources (Last {last_n})",
                        f"{last_df['collected'].mean():.2f}"
                    )
                with col6:
                    st.metric(
                        f"Success % (Last {last_n})",
                        f"{(last_df['success'].mean() * 100):.1f}%"
                    )

            # Baseline comparison if available
            if st.session_state.random_baseline is not None:
                st.markdown("### 🔍 Comparison with Random Policy Baseline")
                base = st.session_state.random_baseline
                colb1, colb2 = st.columns(2)
                with colb1:
                    st.metric(
                        "Avg Reward vs Random",
                        f"{df['reward'].mean():.2f}",
                        delta=f"{df['reward'].mean() - base['avg_reward']:.2f}"
                    )
                with colb2:
                    st.metric(
                        "Avg Resources vs Random",
                        f"{df['collected'].mean():.2f}",
                        delta=f"{df['collected'].mean() - base['avg_resources']:.2f}"
                    )

            st.markdown("### Recent Training History")
            recent_df = df.tail(10)[['episode', 'reward', 'collected', 'success']]
            recent_df['success'] = recent_df['success'].map({1: '✅', 0: '❌'})
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No training data yet. Click 'Train 1 Episode' or 'Train 10 Episodes' to begin!")

    # ----------------- Agent Analysis Tab -----------------
    with tab3:
        st.markdown("### Individual Agent Performance")

        if st.session_state.agents and len(st.session_state.agents) > 0:
            agent_rewards = [agent.total_reward for agent in st.session_state.agents]
            agent_ids = [f'Agent {agent.id}' for agent in st.session_state.agents]

            fig = go.Figure(data=[
                go.Bar(
                    x=agent_ids,
                    y=agent_rewards,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(agent_ids)],
                    text=[f'{r:.1f}' for r in agent_rewards],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Agent Rewards in Current Episode",
                xaxis_title="Agent",
                yaxis_title="Total Reward",
                plot_bgcolor='#1a1a2e',
                paper_bgcolor='#1a1a2e',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Detailed Agent Information")
            agent_details = []
            for agent in st.session_state.agents:
                agent_details.append({
                    'Agent ID': agent.id,
                    'Position': f'({agent.x}, {agent.y})',
                    'Carrying Resource': '✅' if agent.carrying else '❌',
                    'Resource Amount': agent.resource_amount,
                    'Total Reward': f'{agent.total_reward:.2f}',
                    'Color': agent.color
                })
            st.dataframe(pd.DataFrame(agent_details), use_container_width=True)
        else:
            st.info("🎬 No agents available. Click 'Initialize' to create agents.")

        st.markdown("---")
        st.markdown("### Q-Table Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("States Explored", len(st.session_state.q_table))
        with col2:
            total_q_updates = len(st.session_state.q_table) * 6  # 6 actions per state
            st.metric("Total Q-Values", total_q_updates)

        if st.session_state.q_table and len(st.session_state.q_table) > 0:
            st.markdown("### Q-Value Distribution")
            all_q_values = []
            for state_actions in st.session_state.q_table.values():
                all_q_values.extend(state_actions.values())

            if all_q_values:
                fig_dist = go.Figure(data=[go.Histogram(
                    x=all_q_values,
                    nbinsx=30,
                    marker_color='#667eea'
                )])
                fig_dist.update_layout(
                    title="Distribution of Q-Values",
                    xaxis_title="Q-Value",
                    yaxis_title="Frequency",
                    plot_bgcolor='#1a1a2e',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='white'),
                    height=300
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("### Sample Q-Values (First 10 States)")
            sample_states = list(st.session_state.q_table.keys())[:10]
            q_data = []
            for state in sample_states:
                q_values = st.session_state.q_table[state]
                best_action = max(q_values, key=q_values.get)
                q_data.append({
                    'State': state[:40] + '…' if len(state) > 40 else state,
                    'Best Action': best_action,
                    'Q-Value': f"{q_values[best_action]:.3f}",
                    'Avg Q': f"{sum(q_values.values()) / len(q_values):.3f}"
                })
            st.dataframe(pd.DataFrame(q_data), use_container_width=True)

            st.markdown("### Action Preferences Across All States")
            action_counts = defaultdict(int)
            for state_actions in st.session_state.q_table.values():
                best_action = max(state_actions, key=state_actions.get)
                action_counts[best_action] += 1

            if action_counts:
                actions = list(action_counts.keys())
                counts = list(action_counts.values())

                fig_actions = go.Figure(data=[go.Pie(
                    labels=actions,
                    values=counts,
                    hole=0.3,
                    marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F8B500']
                )])
                fig_actions.update_layout(
                    title="Best Action Distribution",
                    plot_bgcolor='#1a1a2e',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='white'),
                    height=400
                )
                st.plotly_chart(fig_actions, use_container_width=True)
        else:
            st.info("🎓 No Q-values learned yet. Train some episodes to populate the Q-table!")

    # ----------------- Documentation Tab -----------------
    with tab4:
        st.markdown("### 📚 Academic Documentation")

        st.markdown(r"""
            # Multi-Agent Reinforcement Learning in Real-Time Strategy

            ## 🎯 Project Overview
            This project implements a multi-agent reinforcement learning system where agents learn to
            collaborate in a resource collection task inside a grid-based RTS environment. The system includes:
            - Multi-agent Q-learning
            - Live environment visualization
            - Step-by-step, 10-episode, and Auto-Train modes
            - Random-policy baseline comparison
            - Q-table analytics
            - Agent-level reward analytics
            - Exploration decay scheduling
            - Full reset & reinitialization logic

            ## 🧠 Algorithm: Q-Learning

            **Q-Learning** is a model-free, off-policy reinforcement learning algorithm that learns the value
            of state–action pairs using the temporal-difference (TD) update rule.

            **Update Rule:**
            ```text
            Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
            ```

            Where:
            - `α` (alpha): Learning rate – determines how strongly new information overwrites old values  
            - `γ` (gamma): Discount factor – importance of long-term rewards  
            - `r`: Reward obtained from taking action `a` in state `s`  
            - `s'`: Next state after taking action  
            - `max_a' Q(s', a')`: Best possible future value

            ## 🏗️ State Representation

            Each agent’s state is compactly encoded as:
            ```
            "x, y, carrying, nearest_res_x, nearest_res_y, base_x, base_y"
            ```
            It includes:
            - Agent position  
            - Whether the agent is carrying a resource  
            - Position of nearest active resource  
            - Base position  

            This representation is efficient and suitable for tabular Q-learning.


            ## 🎮 Actions Available

            Agents choose from:
            - up
            - down
            - left
            - right
            - collect
            - deposit

            Movement is bounded inside the grid. Agents do not block each other (no physical collisions implemented).


            ## 🤝 Multi-Agent Coordination

            Coordination emerges through reward shaping:
            1. **Shared Environment** – Agents operate in a common grid  
            2. **Coordination Bonus** – Agents get +2 reward for every nearby agent while depositing  
            3. **Parallel Learning** – Each agent independently updates its Q-values  
            4. **Emergent Teamwork** – No explicit communication is used  

            ## 🎁 Reward Structure

            | Action / Event        | Reward   | Purpose                          |
            |------------------------|----------|----------------------------------|
            | Collect Resource      | +5       | Encourage resource gathering     |
            | Deposit at Base       | +10      | Reward successful delivery       |
            | Coordination Bonus    | +2/agent | Encourage group behavior         |
            | Movement              | -0.1     | Discourage unnecessary walking   |


            **Coordination bonus**: When an agent deposits at the ally base, it receives an additional +2 per other agent within 3 tiles of the base.
            
            ## 🔄 Episode Structure

            - Each episode runs for **up to 100 steps**  
            - Episode ends early if **8 resources** are deposited  
            - Steps include:  
              - State extraction  
              - Action selection (ε-greedy)  
              - Reward calculation  
              - Q-table update  
              - Optional visualization every 5 steps  

            After each episode:
            - Q-table persists  
            - Episode metrics are recorded  
            - Exploration ε decays  
            - Environment resets for the next episode  

            ## 🔍 Exploration Strategy (ε-greedy with Decay)

            - Higher exploration at the start (`ε₀`)  
            - Decays each episode:  
              ```
              ε ← max(ε_min, ε * ε_decay)
              ```
            - Eventually shifts to exploitation of learned policies  

            ## 🚀 Auto-Training

            The **Auto Train** mode:
            - Runs continuous episodes
            - Updates visualization and Q-values after every episode
            - Uses Streamlit reruns for smooth animation
            - Stops when the user clicks “Stop Auto”

            ## 🎲 Random Policy Baseline

            The system includes a **10-episode random-policy evaluator** to provide baseline metrics:
            - Average reward  
            - Average resources collected  
            
            This allows meaningful comparison with trained policies.

            ## 📊 Performance Metrics Tracked

            1. **Episode Reward Curve**  
            2. **Resources Collected per Episode**  
            3. **Success Rate** (episodes reaching 8 deposits)  
            4. **Cumulative Reward Curve**  
            5. **Q-Table Growth**  
            6. **Action preference distribution**  
            7. **Per-agent reward distribution**

            ## 🤖 Agent Analysis

            For each agent the system tracks:
            - Current Position  
            - Carrying status  
            - Resource amount  
            - Episode reward  
            - Accumulated reward  
            - Assigned color  

            A reward bar chart visualizes per-agent contribution.

            ## 📘 Q-Table Analytics

            The system also displays:
            - **Total states explored**  
            - **Q-value histogram**  
            - **Action preference pie chart**  
            - **Sample of Q-values** (first 10 states)  

            Useful for debugging and understanding learned strategies.

            ## 🌐 Environment Visualization

            The simulation uses Plotly to render:
            - Grid lines  
            - Agents with colors and IDs  
            - Resource nodes with remaining quantity  
            - Ally & enemy bases  
            - Animated updates during training episodes  

            ## 🧱 Initialization & Reset Logic

            - **Initialize**: Creates agents, resources, and bases with randomized resource locations and amounts.
            - **Reset All**: Clears training history, resets Q-table, metrics, epsilon, and re-initializes the environment.
            - Training buttons will automatically initialize the environment if it hasn't been initialized yet.

            ## ⚙️ Sidebar Controls

            The user controls:
            - Number of agents  
            - Grid size  
            - Learning rate (α)  
            - Discount Factor (γ)  
            - ε-start, ε-minimum, ε-decay  
            - Episode controls (Initialize, Train 1, Train 10, Auto Train, Reset All)

            ## 💡 Usage Tips

            1. Start with **ε₀ ≈ 0.3**, `ε_min ≈ 0.05`, `decay ≈ 0.99`  
            2. Train **30–50 episodes** for learning to stabilize  
            3. Increase grid size for complex environments  
            4. Analyze Q-table patterns in Agent Analysis tab  
            5. Compare trained agents with the **random baseline**  

            ## 📎 Limitations

            - Agents do not have explicit communication channels; coordination emerges only via reward shaping.
            - Tabular Q-learning does not scale to large grids. Consider function approximation (DQN/actor-critic) for larger environments.  
            - No collision physics: multiple agents can occupy the same cell.
            - State representation is intentionally compact; richer states (e.g., full observation matrices) can be used for function approximation. 
            - Agents do not observe each other directly (only indirectly through rewards)

            ## 📝 Conclusion

            This project demonstrates multi-agent reinforcement learning in a structured RTS environment.
            Through shared rewards, exploration scheduling, and repeated training, agents learn to show
            **emergent cooperative behavior**. The system includes rich analytics and visualizations
            ideal for academic demonstration and experimentation.
            """)

        st.markdown("---")
        st.success("💡 Tip: Run 50+ episodes and compare with the random baseline to show clear learning gains in your report!")


if __name__ == "__main__":
    main()

