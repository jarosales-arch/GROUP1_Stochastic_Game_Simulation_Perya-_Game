import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Perya Game Simulation",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved design
st.markdown("""
<style>
    /* Main styling */
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9F43 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #2E4053;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #FF4B4B;
        font-weight: 600;
    }
    
    /* Cards and containers */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #FF4B4B;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .info-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #FF9800;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9F43 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2E4053 0%, #1C2833 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 10px 10px 0 0;
        gap: 1rem;
        padding: 1rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9F43 100%);
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-fair {
        background-color: #4CAF50;
        color: white;
    }
    
    .badge-tweaked {
        background-color: #FF9800;
        color: white;
    }
    
    /* Color chips */
    .color-chip {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
        border: 2px solid rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Game logic functions (UNCHANGED - keeping original functionality)
class ColorGame:
    def __init__(self, game_type="Fair Game", house_edge=5.0):
        self.colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
        self.game_type = game_type
        self.house_edge = house_edge / 100
        
        if game_type == "Fair Game":
            self.setup_fair_game()
        else:
            self.setup_tweaked_game()
    
    def setup_fair_game(self):
        self.probabilities = [1/6] * 6
        self.payout_multiplier = 5.0
    
    def setup_tweaked_game(self):
        base_prob = 1/6
        reduction = self.house_edge / 6
        
        self.probabilities = []
        for i in range(6):
            if i < 3:
                prob = base_prob - reduction
            else:
                prob = base_prob + reduction
            self.probabilities.append(prob)
        
        total = sum(self.probabilities)
        self.probabilities = [p/total for p in self.probabilities]
        self.payout_multiplier = 4.8
    
    def play_round(self, player_color, bet_amount):
        winning_color = np.random.choice(self.colors, p=self.probabilities)
        
        if player_color == winning_color:
            win_amount = bet_amount * self.payout_multiplier
            return True, win_amount, winning_color
        else:
            return False, -bet_amount, winning_color

def run_monte_carlo_simulation(game_type, num_simulations, bet_amount, initial_balance, player_color):
    if game_type == "Compare Both":
        results = {}
        for gtype in ["Fair Game", "Tweaked Game"]:
            game = ColorGame(gtype)
            balance = initial_balance
            history = []
            wins = 0
            win_history = []
            winning_colors = []
            
            for i in range(num_simulations):
                if balance < bet_amount:
                    history.append(balance)
                    win_history.append(False)
                    winning_colors.append(None)
                    continue
                    
                won, amount, winning_color = game.play_round(player_color, bet_amount)
                balance += amount
                history.append(balance)
                win_history.append(won)
                winning_colors.append(winning_color)
                
                if won:
                    wins += 1
            
            results[gtype] = {
                'history': history,
                'final_balance': balance,
                'wins': wins,
                'total_games': num_simulations,
                'win_rate': wins / num_simulations,
                'total_profit': balance - initial_balance,
                'expected_value': (wins * bet_amount * game.payout_multiplier - 
                                 (num_simulations - wins) * bet_amount) / num_simulations,
                'win_history': win_history,
                'winning_colors': winning_colors,
                'probabilities': game.probabilities,
                'payout_multiplier': game.payout_multiplier
            }
        return results
    
    else:
        game = ColorGame(game_type)
        balance = initial_balance
        history = []
        wins = 0
        win_history = []
        winning_colors = []
        
        for i in range(num_simulations):
            if balance < bet_amount:
                history.append(balance)
                win_history.append(False)
                winning_colors.append(None)
                continue
                
            won, amount, winning_color = game.play_round(player_color, bet_amount)
            balance += amount
            history.append(balance)
            win_history.append(won)
            winning_colors.append(winning_color)
            
            if won:
                wins += 1
        
        return {
            'history': history,
            'final_balance': balance,
            'wins': wins,
            'total_games': num_simulations,
            'win_rate': wins / num_simulations,
            'total_profit': balance - initial_balance,
            'expected_value': (wins * bet_amount * game.payout_multiplier - 
                             (num_simulations - wins) * bet_amount) / num_simulations,
            'win_history': win_history,
            'winning_colors': winning_colors,
            'probabilities': game.probabilities,
            'payout_multiplier': game.payout_multiplier
        }

# Header with enhanced design
st.markdown("<h1 class='main-title'>🎲 FILIPINO PERYA COLOR GAME SIMULATOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Modeling and Simulation of Casino Games with House Edge Analysis | CSEC 413 Final Project</p>", unsafe_allow_html=True)

# Sidebar with improved design
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    st.markdown("### 🎮 CONTROL PANEL")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Game parameters in expandable sections
    with st.expander("⚙️ SIMULATION PARAMETERS", expanded=True):
        num_simulations = st.slider(
            "Number of Games", 
            1000, 50000, 10000, 1000,
            help="More games = more accurate results but slower simulation"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            bet_amount = st.number_input(
                "Bet Amount (₱)", 
                min_value=1.0, max_value=1000.0, value=10.0, step=10.0,
                help="Amount bet on each game"
            )
        with col2:
            initial_balance = st.number_input(
                "Initial Balance (₱)", 
                min_value=100.0, max_value=10000.0, value=1000.0, step=100.0,
                help="Starting amount of money"
            )
    
    with st.expander("🎯 GAME SETTINGS", expanded=True):
        game_type = st.radio(
            "Game Type:",
            ["Fair Game", "Tweaked Game", "Compare Both"],
            index=2,
            help="Fair: Equal probabilities | Tweaked: House advantage | Compare: Side-by-side analysis"
        )
        
        # Color selection with visual chips
        colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
        color_display = []
        for color in colors:
            color_code = {
                "Red": "#FF4B4B",
                "Green": "#4CAF50",
                "Blue": "#2196F3",
                "Yellow": "#FFD700",
                "White": "#FFFFFF",
                "Black": "#000000"
            }
            color_display.append(f"<span class='color-chip' style='background-color:{color_code[color]};'></span>{color}")
        
        player_color = st.selectbox(
            "Your Color:",
            colors,
            format_func=lambda x: f"<span class='color-chip' style='background-color:{color_code[x]};'></span>{x}",
            help="Choose which color to bet on"
        )
    
    # Advanced settings
    with st.expander("🔧 ADVANCED SETTINGS", expanded=False):
        if game_type == "Tweaked Game" or game_type == "Compare Both":
            house_edge = st.slider(
                "House Edge (%)", 
                1.0, 20.0, 5.0, 0.5,
                help="Casino's mathematical advantage over players"
            )
            tweak_type = st.selectbox(
                "Tweak Method:", 
                ["Weighted Probabilities", "Modified Payouts", "Both"],
                help="How the house edge is implemented"
            )
        
        # Add animation toggle
        enable_animations = st.checkbox("Enable Animations", value=True)
        show_advanced_stats = st.checkbox("Show Advanced Statistics", value=False)
    
    # Run button with better styling
    st.markdown("<br>", unsafe_allow_html=True)
    run_simulation = st.button(
        "🚀 RUN MONTE CARLO SIMULATION", 
        type="primary",
        use_container_width=True
    )
    
    # Quick stats in sidebar
    st.markdown("---")
    st.markdown("### 📊 QUICK STATS")
    
    col1, col2 = st.columns(2)
    with col1:
        fair_ev = round((1/6) * bet_amount * 5 - (5/6) * bet_amount, 2)
        st.metric("Fair EV", f"₱{fair_ev}")
    
    with col2:
        if game_type != "Fair Game":
            tweaked_ev = round(fair_ev * 0.95, 2)
            st.metric("Tweaked EV", f"₱{tweaked_ev}")
    
    # Information
    st.markdown("---")
    with st.expander("ℹ️ ABOUT THIS SIMULATION"):
        st.info("""
        *Monte Carlo Simulation* uses random sampling to model probabilistic systems.
        
        *House Edge* represents the casino's long-term advantage.
        
        *Expected Value (EV)* is the average outcome per game.
        
        Results may vary due to random sampling.
        """)

# Main content
if run_simulation:
    # Progress animation
    with st.spinner('🎲 Shuffling probabilities...'):
        time.sleep(0.5)
    
    # Progress bar with better styling
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create placeholder for simulation animation
    simulation_placeholder = st.empty()
    
    # Run simulation with progress updates
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        status_text.text(f"🎯 Running simulation... {i+1}% complete")
        
        # Update animation placeholder
        if enable_animations and i % 20 == 0:
            with simulation_placeholder.container():
                # Create a simple animation of rolling dice
                dice_faces = ['⚀', '⚁', '⚂', '⚃', '⚄', '⚅']
                current_dice = dice_faces[i // 20 % 6]
                st.markdown(f"<h3 style='text-align: center; font-size: 4rem;'>{current_dice}</h3>", unsafe_allow_html=True)
    
    status_text.success("✅ Simulation complete!")
    simulation_placeholder.empty()
    
    # Run the simulation
    if game_type == "Compare Both":
        results = run_monte_carlo_simulation(
            game_type, num_simulations, bet_amount, 
            initial_balance, player_color
        )
        
        # Comparison header
        st.markdown("<h2 class='section-header'>📊 COMPARATIVE ANALYSIS</h2>", unsafe_allow_html=True)
        
        # Create two columns for comparison with badges
        col1, col2 = st.columns(2)
        
        for idx, (gtype, result) in enumerate(results.items()):
            with col1 if idx == 0 else col2:
                # Header with badge
                badge_class = "badge-fair" if gtype == "Fair Game" else "badge-tweaked"
                st.markdown(f"<h3><span class='badge {badge_class}'>{gtype.upper()}</span></h3>", unsafe_allow_html=True)
                
                # Metrics in cards
                with st.container():
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric(
                            "Final Balance", 
                            f"₱{result['final_balance']:,.2f}",
                            delta=f"₱{result['total_profit']:,.2f}" if result['total_profit'] >= 0 else f"₱{result['total_profit']:,.2f}",
                            delta_color="normal"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Win Rate", f"{result['win_rate']*100:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col_metric2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Total Games", f"{result['total_games']:,}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        ev_color = "🟢" if result['expected_value'] > 0 else "🔴"
                        st.metric("Expected Value", f"₱{result['expected_value']:.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Plot balance history
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=result['history'],
                    mode='lines',
                    name='Balance',
                    line=dict(
                        color='#4CAF50' if result['final_balance'] >= initial_balance else '#FF4B4B',
                        width=3
                    ),
                    fill='tozeroy',
                    fillcolor='rgba(76, 175, 80, 0.1)' if result['final_balance'] >= initial_balance else 'rgba(255, 75, 75, 0.1)'
                ))
                fig.add_hline(
                    y=initial_balance, 
                    line_dash="dash", 
                    line_color="#666",
                    annotation_text="Initial Balance",
                    annotation_position="bottom right"
                )
                fig.update_layout(
                    title=f"Balance Progression - {gtype}",
                    xaxis_title="Game Number",
                    yaxis_title="Balance (₱)",
                    height=350,
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("<h3 class='section-header'>📈 DETAILED COMPARISON</h3>", unsafe_allow_html=True)
        
        fair_result = results["Fair Game"]
        tweaked_result = results["Tweaked Game"]
        
        comparison_data = {
            "Metric": ["Final Balance", "Total Profit", "Win Rate", "Expected Value", 
                      "Total Wins", "Profit/Loss Ratio", "Max Balance", "Min Balance"],
            "Fair Game": [
                fair_result['final_balance'],
                fair_result['total_profit'],
                fair_result['win_rate'] * 100,
                fair_result['expected_value'],
                fair_result['wins'],
                fair_result['total_profit'] / initial_balance,
                max(fair_result['history']),
                min(fair_result['history'])
            ],
            "Tweaked Game": [
                tweaked_result['final_balance'],
                tweaked_result['total_profit'],
                tweaked_result['win_rate'] * 100,
                tweaked_result['expected_value'],
                tweaked_result['wins'],
                tweaked_result['total_profit'] / initial_balance,
                max(tweaked_result['history']),
                min(tweaked_result['history'])
            ],
            "Difference": [
                tweaked_result['final_balance'] - fair_result['final_balance'],
                tweaked_result['total_profit'] - fair_result['total_profit'],
                (tweaked_result['win_rate'] - fair_result['win_rate']) * 100,
                tweaked_result['expected_value'] - fair_result['expected_value'],
                tweaked_result['wins'] - fair_result['wins'],
                (tweaked_result['total_profit'] / initial_balance) - (fair_result['total_profit'] / initial_balance),
                max(tweaked_result['history']) - max(fair_result['history']),
                min(tweaked_result['history']) - min(fair_result['history'])
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Style the comparison table
        def color_diff(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}; font-weight: bold;'
        
        st.dataframe(
            df_comparison.style.format({
                "Fair Game": "{:,.2f}",
                "Tweaked Game": "{:,.2f}",
                "Difference": "{:+,.2f}"
            }).applymap(color_diff, subset=['Difference']),
            use_container_width=True
        )
        
        # Download option for comparison results
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### 📥 DOWNLOAD RESULTS")
        
        all_data = []
        for gtype, result in results.items():
            n = len(result['history'])
            for i in range(n):
                all_data.append({
                    'Game_Type': gtype,
                    'Game_Number': i + 1,
                    'Balance': result['history'][i] if i < len(result['history']) else None,
                    'Win': result['win_history'][i] if i < len(result['win_history']) else None,
                    'Winning_Color': result['winning_colors'][i] if i < len(result['winning_colors']) else None
                })
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📊 Download All Simulation Data (CSV)",
                data=csv,
                file_name="perya_color_game_comparison.csv",
                mime="text/csv",
                help="Download complete simulation data for further analysis"
            )
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # Single game type results
        result = run_monte_carlo_simulation(
            game_type, num_simulations, bet_amount, 
            initial_balance, player_color
        )
        
        # Header with game type badge
        badge_type = "badge-fair" if game_type == "Fair Game" else "badge-tweaked"
        st.markdown(f"<h2 class='section-header'><span class='badge {badge_type}'>{game_type.upper()}</span> SIMULATION RESULTS</h2>", unsafe_allow_html=True)
        
        # Key metrics in a grid
        st.markdown("### 📈 KEY PERFORMANCE INDICATORS")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            delta_color = "normal" if result['total_profit'] >= 0 else "inverse"
            st.metric(
                "Final Balance", 
                f"₱{result['final_balance']:,.2f}",
                delta=f"₱{result['total_profit']:,.2f}",
                delta_color=delta_color
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                "Win Rate", 
                f"{result['win_rate']*100:.2f}%",
                delta=f"{result['wins']:,} wins"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Games", f"{result['total_games']:,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            ev_icon = "📈" if result['expected_value'] > 0 else "📉"
            st.metric(
                "Expected Value", 
                f"₱{result['expected_value']:.3f}",
                delta=f"{ev_icon} Per Game"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Balance History", 
            "🎯 Win Analysis", 
            "⚖️ Probability Analysis", 
            "📈 Statistics"
        ])
        
        with tab1:
            # Main balance chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Balance Progression', 'Rolling Average (100 games)'),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            # Balance line
            fig.add_trace(
                go.Scatter(
                    y=result['history'],
                    mode='lines',
                    name='Balance',
                    line=dict(color='#2196F3', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(33, 150, 243, 0.1)'
                ),
                row=1, col=1
            )
            
            # Initial balance line
            fig.add_hline(
                y=initial_balance,
                line_dash="dash",
                line_color="#FF9800",
                annotation_text="Initial Balance",
                annotation_position="bottom right",
                row=1, col=1
            )
            
            # Rolling average
            window_size = min(100, num_simulations // 10)
            if window_size > 1:
                rolling_avg = pd.Series(result['history']).rolling(window=window_size).mean()
                fig.add_trace(
                    go.Scatter(
                        y=rolling_avg,
                        mode='lines',
                        name=f'Rolling Avg ({window_size} games)',
                        line=dict(color='#4CAF50', width=2)
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                template="plotly_white",
                hovermode="x unified"
            )
            
            fig.update_xaxes(title_text="Game Number", row=2, col=1)
            fig.update_yaxes(title_text="Balance (₱)", row=1, col=1)
            fig.update_yaxes(title_text="Average Balance (₱)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            if result['final_balance'] < initial_balance * 0.5:
                st.warning("⚠️ *Warning:* Player lost more than 50% of initial balance. Consider smaller bet sizes.")
            elif result['final_balance'] > initial_balance * 1.5:
                st.success("🎉 *Great performance!* Player gained more than 50% profit.")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Win/loss pie chart
                win_loss_counts = pd.Series(result['win_history']).value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Losses', 'Wins'],
                    values=[win_loss_counts.get(False, 0), win_loss_counts.get(True, 0)],
                    hole=.4,
                    marker_colors=['#FF4B4B', '#4CAF50'],
                    textinfo='percent+value',
                    textposition='inside'
                )])
                fig.update_layout(
                    title="Win/Loss Distribution",
                    height=400,
                    annotations=[dict(
                        text=f"{result['win_rate']*100:.1f}%",
                        x=0.5, y=0.5,
                        font_size=20,
                        showarrow=False
                    )]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Winning colors distribution
                if 'winning_colors' in result:
                    color_counts = pd.Series(result['winning_colors']).value_counts()
                    color_palette = {
                        "Red": "#FF4B4B",
                        "Green": "#4CAF50",
                        "Blue": "#2196F3",
                        "Yellow": "#FFD700",
                        "White": "#666666",
                        "Black": "#000000"
                    }
                    
                    fig = px.bar(
                        x=color_counts.index,
                        y=color_counts.values,
                        title="Winning Colors Frequency",
                        labels={'x': 'Color', 'y': 'Number of Wins'},
                        color=color_counts.index,
                        color_discrete_map=color_palette
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        xaxis_title="Color",
                        yaxis_title="Frequency"
                    )
                    
                    # Highlight player's color
                    if player_color in color_counts.index:
                        fig.update_traces(
                            marker_line_width=3,
                            marker_line_color='black',
                            selector=dict(name=player_color)
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Probability analysis with enhanced visualization
            if 'probabilities' in result:
                colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Probability Distribution', 'Fair vs Actual Comparison'),
                    specs=[[{'type': 'pie'}, {'type': 'bar'}]]
                )
                
                # Pie chart
                fig.add_trace(
                    go.Pie(
                        labels=colors,
                        values=result['probabilities'],
                        hole=.3,
                        marker_colors=['#FF4B4B', '#4CAF50', '#2196F3', '#FFD700', '#666666', '#000000'],
                        textinfo='percent+label',
                        textposition='inside'
                    ),
                    row=1, col=1
                )
                
                # Bar chart comparison
                fig.add_trace(
                    go.Bar(
                        x=colors,
                        y=[1/6] * 6,
                        name='Fair Probability',
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=colors,
                        y=result['probabilities'],
                        name=f'{game_type} Probability',
                        marker_color='coral',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=500,
                    barmode='group',
                    showlegend=True,
                    template="plotly_white"
                )
                
                fig.update_xaxes(title_text="Color", row=1, col=2)
                fig.update_yaxes(title_text="Probability", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Game information
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                    st.markdown("#### 🎰 Game Information")
                    st.markdown(f"*Payout Multiplier:* {result['payout_multiplier']}:1")
                    st.markdown(f"*Player's Color:* <span class='color-chip' style='background-color:{color_palette[player_color]};'></span>{player_color}", unsafe_allow_html=True)
                    st.markdown(f"*Bet Amount:* ₱{bet_amount:,.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    if game_type == "Tweaked Game":
                        st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
                        st.markdown("#### ⚠️ House Edge Analysis")
                        fair_ev = (1/6) * bet_amount * 5 - (5/6) * bet_amount
                        tweaked_ev = result['expected_value']
                        house_edge = (fair_ev - tweaked_ev) / bet_amount * 100
                        st.markdown(f"*Calculated House Edge:* {house_edge:.2f}%")
                        st.markdown(f"*Player EV:* ₱{tweaked_ev:.3f} per game")
                        st.markdown(f"*Fair EV:* ₱{fair_ev:.3f} per game")
                        st.markdown("</div>", unsafe_allow_html=True)
        
        with tab4:
            # Statistical summary with enhanced metrics
            st.markdown("### 📊 Statistical Summary")
            
            history_series = pd.Series(result['history'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("##### Balance Statistics")
                stats_data = {
                    'Statistic': ['Mean Balance', 'Median Balance', 'Std Deviation', 
                                'Minimum Balance', 'Maximum Balance', 'Range', 
                                'Interquartile Range (IQR)', 'Coefficient of Variation'],
                    'Value': [
                        history_series.mean(),
                        history_series.median(),
                        history_series.std(),
                        history_series.min(),
                        history_series.max(),
                        history_series.max() - history_series.min(),
                        history_series.quantile(0.75) - history_series.quantile(0.25),
                        (history_series.std() / history_series.mean()) * 100 if history_series.mean() != 0 else 0
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(
                    stats_df.style.format({'Value': '{:,.2f}'}),
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("##### Performance Metrics")
                
                # Risk metrics
                max_drawdown = (history_series.min() - initial_balance) / initial_balance * 100
                volatility = history_series.std()
                
                metrics_data = {
                    'Metric': ['Total Return (₱)', 'Return (%)', 'Sharpe Ratio*', 
                              'Max Drawdown (%)', 'Volatility (₱)', 'Risk of Ruin*',
                              'Profit Factor', 'Recovery Factor*'],
                    'Value': [
                        result['total_profit'],
                        result['total_profit'] / initial_balance * 100,
                        result['total_profit'] / volatility if volatility > 0 else 0,
                        max_drawdown,
                        volatility,
                        "High" if result['final_balance'] < bet_amount else "Low" if result['final_balance'] < initial_balance * 0.5 else "Medium",
                        abs(result['wins'] * bet_amount * result['payout_multiplier']) / abs((result['total_games'] - result['wins']) * bet_amount) if result['total_games'] > result['wins'] else float('inf'),
                        result['total_profit'] / abs(max_drawdown/100 * initial_balance) if max_drawdown < 0 else float('inf')
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                
                # Format values
                def format_metric(val):
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        if abs(val) > 1000:
                            return f"{val:,.0f}"
                        elif abs(val) > 1:
                            return f"{val:,.2f}"
                        else:
                            return f"{val:.4f}"
                    return str(val)
                
                metrics_df['Value'] = metrics_df['Value'].apply(format_metric)
                st.dataframe(metrics_df, use_container_width=True)
                st.caption("*Approximate calculations for educational purposes")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Distribution plot
            st.markdown("##### Balance Distribution")
            fig = px.histogram(
                history_series,
                nbins=50,
                title="Balance Distribution Histogram",
                labels={'value': 'Balance (₱)', 'count': 'Frequency'}
            )
            fig.add_vline(
                x=initial_balance,
                line_dash="dash",
                line_color="red",
                annotation_text="Initial Balance"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Download results for single game
    if game_type != "Compare Both":
        actual_games = len(result['history'])
        
        # Create result DataFrame
        game_numbers = list(range(1, actual_games + 1))
        balances = result['history']
        
        win_history = result.get('win_history', [None] * actual_games)
        if len(win_history) < actual_games:
            win_history.extend([None] * (actual_games - len(win_history)))
        elif len(win_history) > actual_games:
            win_history = win_history[:actual_games]
            
        winning_colors = result.get('winning_colors', [None] * actual_games)
        if len(winning_colors) < actual_games:
            winning_colors.extend([None] * (actual_games - len(winning_colors)))
        elif len(winning_colors) > actual_games:
            winning_colors = winning_colors[:actual_games]
        
        result_df = pd.DataFrame({
            'Game_Number': game_numbers[:actual_games],
            'Balance': balances[:actual_games],
            'Win': win_history[:actual_games],
            'Winning_Color': winning_colors[:actual_games],
            'Cumulative_Profit': [bal - initial_balance for bal in balances[:actual_games]]
        })
        
        csv = result_df.to_csv(index=False)
        
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### 📥 DOWNLOAD SIMULATION DATA")
        st.download_button(
            label=f"📊 Download {game_type} Data (CSV)",
            data=csv,
            file_name=f"perya_{game_type.replace(' ', '_').lower()}_simulation.csv",
            mime="text/csv",
            help="Download complete simulation data for further analysis"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Simulation insights with enhanced presentation
    st.markdown("<h2 class='section-header'>🧠 SIMULATION INSIGHTS & KEY TAKEAWAYS</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Mathematical Principles")
        st.markdown("""
        *Law of Large Numbers*: 
        As simulation size increases, results converge to expected values.
        
        *House Edge Impact*: 
        Even 1-2% edge significantly affects long-term profitability.
        
        *Expected Value (EV)*:
        Average outcome per game determines long-term results.
        
        *Monte Carlo Method*:
        Random sampling approximates complex probabilistic systems.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚠️ Practical Implications")
        st.markdown("""
        *Bankroll Management*:
        Bet size relative to balance is crucial for survival.
        
        *Short-term Variance*:
        Results can deviate significantly from EV in small samples.
        
        *Casino Profitability*:
        House edge ensures long-term profitability despite volatility.
        
        *Player Psychology*:
        Short-term wins can mask long-term disadvantage.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Default view - Landing page with enhanced design
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #2E4053; margin-bottom: 2rem;'>🎯 PROJECT OVERVIEW & INTRODUCTION</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Introduction cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🎮 The Game")
        st.markdown("""
        Traditional Filipino "Perya" color game
        - 6 colors to bet on
        - 5:1 payout for correct guess
        - Random selection mechanism
        - Simple yet profound probability model
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🔬 The Science")
        st.markdown("""
        Monte Carlo Simulation
        - 10,000+ game simulations
        - Statistical analysis
        - Probability modeling
        - Stochastic process visualization
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 The Purpose")
        st.markdown("""
        Educational Demonstration
        - Understand house edge
        - Analyze risk vs reward
        - Visualize probability
        - Learn statistical modeling
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # How to use section
    st.markdown("<h3 class='section-header'>🚀 GETTING STARTED</h3>", unsafe_allow_html=True)
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.markdown("""
        ### 📋 Quick Start Guide
        
        1. *Adjust Parameters* in sidebar
        2. *Select Game Type* (Fair/Tweaked/Compare)
        3. *Choose Your Color* to bet on
        4. *Click RUN SIMULATION* button
        5. *Analyze Results* in real-time
        
        ### ⚙️ Recommended Settings
        
        - *Beginner*: 1,000 games, Fair Game
        - *Intermediate*: 10,000 games, Compare Both
        - *Advanced*: 50,000 games, Tweaked Game
        
        ### 🎯 Tips for Learning
        
        Start with small simulations to understand patterns, then scale up for more accurate results.
        """)
    
    with steps_col2:
        # Interactive example visualization
        st.markdown("#### 📊 Example Visualizations")
        
        tab1, tab2 = st.tabs(["Fair Game", "Tweaked Game"])
        
        with tab1:
            fair_probs = [1/6] * 6
            fig = go.Figure(data=[go.Pie(
                labels=['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black'],
                values=fair_probs,
                hole=.4,
                marker_colors=['#FF4B4B', '#4CAF50', '#2196F3', '#FFD700', '#666666', '#000000']
            )])
            fig.update_layout(
                title="Fair Game: Equal Probabilities",
                height=300,
                annotations=[dict(
                    text="16.67% each",
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            tweaked_probs = [0.14, 0.14, 0.14, 0.19, 0.19, 0.20]
            fig = go.Figure(data=[go.Pie(
                labels=['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black'],
                values=tweaked_probs,
                hole=.4,
                marker_colors=['#FF4B4B', '#4CAF50', '#2196F3', '#FFD700', '#666666', '#000000']
            )])
            fig.update_layout(
                title="Tweaked Game: Weighted Probabilities",
                height=300,
                annotations=[dict(
                    text="~5% House Edge",
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Learning outcomes
    st.markdown("<h3 class='section-header'>🎓 LEARNING OUTCOMES</h3>", unsafe_allow_html=True)
    
    outcomes = """
    #### 📊 What You'll Learn:
    
    1. *Probability Theory*: Understanding of independent events and expected value
    2. *Statistical Analysis*: Interpretation of simulation results and confidence intervals
    3. *Casino Mathematics*: How house edge affects long-term profitability
    4. *Monte Carlo Methods*: Application of computational statistics
    5. *Risk Management*: Bankroll management and bet sizing strategies
    6. *Data Visualization*: Effective presentation of statistical data
    
    #### 🔍 Key Concepts Covered:
    
    - *Expected Value (EV)*: Mathematical expectation per game
    - *House Edge*: Casino's mathematical advantage
    - *Law of Large Numbers*: Convergence to theoretical probabilities
    - *Volatility*: Short-term vs long-term results
    - *Risk of Ruin*: Probability of losing entire bankroll
    
    #### 💡 Real-world Applications:
    
    - Casino game design and analysis
    - Financial risk modeling
    - Insurance probability calculations
    - Game theory applications
    - Statistical quality control
    """
    
    st.markdown(outcomes)

# Enhanced Footer
st.markdown("""
<div class='footer'>
    <h3>🎓 CSEC 413 - Modeling and Simulation</h3>
    <p><strong>Final Project: Stochastic Game Simulation & Analysis</strong></p>
    <p>This educational tool demonstrates mathematical concepts behind casino games.</p>
    <p style='color: #FF4B4B; font-weight: bold;'>⚠️ Gambling involves significant risk. This simulation is for educational purposes only.</p>
    <p>© 2024 Filipino Perya Game Simulation | Made with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)