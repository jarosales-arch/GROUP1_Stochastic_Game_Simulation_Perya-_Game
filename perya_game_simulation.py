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
    
    /* Color display fix */
    .color-option {
        display: flex;
        align-items: center;
        padding: 5px 0;
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

# Enhanced Game logic functions
class EnhancedColorGame:
    def __init__(self, game_type="Fair Game", tweak_method="Weighted Probabilities", 
                 house_edge=5.0, payout_reduction=0.0):
        self.colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
        self.game_type = game_type
        self.tweak_method = tweak_method
        self.house_edge = house_edge / 100
        self.payout_reduction = payout_reduction / 100
        
        if game_type == "Fair Game":
            self.setup_fair_game()
        else:
            if tweak_method == "Weighted Probabilities":
                self.setup_weighted_probabilities()
            elif tweak_method == "Modified Payouts":
                self.setup_modified_payouts()
            elif tweak_method == "Normal Distribution":
                self.setup_normal_distribution()
            elif tweak_method == "Both":
                self.setup_both_tweaks()
            else:
                self.setup_fair_game()
    
    def setup_fair_game(self):
        """Setup fair game with equal probabilities"""
        self.probabilities = [1/6] * 6
        self.payout_multiplier = 5.0
        self.tweak_description = "Fair Game: Equal probabilities for all colors (16.67% each)"
    
    def setup_weighted_probabilities(self):
        """Setup game with weighted probabilities"""
        base_prob = 1/6
        reduction = self.house_edge / 12  # Split edge between player and house colors
        
        # Player colors (first 3) have lower probability
        # House colors (last 3) have higher probability
        self.probabilities = []
        for i in range(6):
            if i < 3:  # Player colors
                prob = base_prob - reduction
            else:  # House colors
                prob = base_prob + reduction
            self.probabilities.append(max(0.01, prob))  # Ensure minimum probability
        
        total = sum(self.probabilities)
        self.probabilities = [p/total for p in self.probabilities]
        self.payout_multiplier = 5.0  # Fair payout
        self.tweak_description = f"Weighted Probabilities: Player colors have {self.house_edge*100:.1f}% lower probability"
    
    def setup_modified_payouts(self):
        """Setup game with modified payouts"""
        self.probabilities = [1/6] * 6  # Fair probabilities
        # Reduce payout to create house edge
        reduction_factor = 1 - self.house_edge
        self.payout_multiplier = 5.0 * reduction_factor
        self.tweak_description = f"Modified Payouts: Payout reduced from 5.0 to {self.payout_multiplier:.2f} ({self.house_edge*100:.1f}% edge)"
    
    def setup_normal_distribution(self):
        """Setup game with normal distribution payouts"""
        self.probabilities = [1/6] * 6  # Fair probabilities
        
        # Create normal distribution for payouts with negative mean
        # Mean payout will be slightly negative to create house edge
        self.use_normal_distribution = True
        self.payout_mean = -self.house_edge * 10  # Negative mean for house edge
        self.payout_std = 15.0  # Standard deviation for variance
        
        # Base multiplier for display purposes
        self.payout_multiplier = 5.0
        self.tweak_description = f"Normal Distribution: Payouts ~N(μ={self.payout_mean:.2f}, σ={self.payout_std:.1f}) with {self.house_edge*100:.1f}% house edge"
    
    def setup_both_tweaks(self):
        """Setup game with both probability and payout tweaks"""
        # Apply probability tweak
        base_prob = 1/6
        prob_reduction = self.house_edge / 24  # Half the edge for probabilities
        
        self.probabilities = []
        for i in range(6):
            if i < 3:  # Player colors
                prob = base_prob - prob_reduction
            else:  # House colors
                prob = base_prob + prob_reduction
            self.probabilities.append(max(0.01, prob))
        
        total = sum(self.probabilities)
        self.probabilities = [p/total for p in self.probabilities]
        
        # Apply payout tweak (half the edge)
        payout_reduction = self.house_edge / 2
        reduction_factor = 1 - payout_reduction
        self.payout_multiplier = 5.0 * reduction_factor
        
        self.tweak_description = f"Both Tweaks: Probabilities weighted + Payout reduced to {self.payout_multiplier:.2f} (Total {self.house_edge*100:.1f}% edge)"
        self.use_normal_distribution = False
    
    def play_round(self, player_color, bet_amount):
        """Play one round of the game"""
        winning_color = np.random.choice(self.colors, p=self.probabilities)
        
        if player_color == winning_color:
            if hasattr(self, 'use_normal_distribution') and self.use_normal_distribution:
                # Use normal distribution for payout
                win_amount = max(0, np.random.normal(
                    bet_amount * self.payout_mean + bet_amount * self.payout_multiplier,
                    bet_amount * self.payout_std
                ))
            else:
                win_amount = bet_amount * self.payout_multiplier
            return True, win_amount, winning_color
        else:
            return False, -bet_amount, winning_color
    
    def calculate_theoretical_ev(self, bet_amount):
        """Calculate theoretical expected value"""
        if self.game_type == "Fair Game":
            return (1/6) * bet_amount * 5.0 - (5/6) * bet_amount
        
        player_colors_idx = [i for i, color in enumerate(self.colors) if color in ["Red", "Green", "Blue"]]
        win_prob = sum([self.probabilities[i] for i in player_colors_idx])
        
        if hasattr(self, 'use_normal_distribution') and self.use_normal_distribution:
            # For normal distribution, approximate EV
            return win_prob * (bet_amount * self.payout_mean + bet_amount * self.payout_multiplier) - (1 - win_prob) * bet_amount
        else:
            return win_prob * bet_amount * self.payout_multiplier - (1 - win_prob) * bet_amount
    
    def calculate_house_edge(self, bet_amount):
        """Calculate theoretical house edge"""
        fair_ev = (1/6) * bet_amount * 5.0 - (5/6) * bet_amount
        current_ev = self.calculate_theoretical_ev(bet_amount)
        return (fair_ev - current_ev) / bet_amount * 100

def run_monte_carlo_simulation(game_type, num_simulations, bet_amount, initial_balance, 
                              player_color, tweak_method="Weighted Probabilities", 
                              house_edge=5.0):
    """Run Monte Carlo simulation for the game"""
    
    def simulate_game(game_obj, num_sims, bet_amt, init_bal, player_col):
        balance = init_bal
        history = []
        wins = 0
        win_history = []
        winning_colors = []
        all_payouts = []
        
        for i in range(num_sims):
            if balance < bet_amt:
                history.append(balance)
                win_history.append(False)
                winning_colors.append(None)
                all_payouts.append(0)
                continue
                
            won, amount, winning_color = game_obj.play_round(player_col, bet_amt)
            balance += amount
            history.append(balance)
            win_history.append(won)
            winning_colors.append(winning_color)
            all_payouts.append(amount if won else -bet_amt)
            
            if won:
                wins += 1
        
        return {
            'history': history,
            'final_balance': balance,
            'wins': wins,
            'total_games': num_sims,
            'win_rate': wins / num_sims,
            'total_profit': balance - init_bal,
            'expected_value': np.mean(all_payouts),
            'win_history': win_history,
            'winning_colors': winning_colors,
            'probabilities': game_obj.probabilities,
            'payout_multiplier': game_obj.payout_multiplier,
            'all_payouts': all_payouts,
            'tweak_description': game_obj.tweak_description,
            'theoretical_ev': game_obj.calculate_theoretical_ev(bet_amt),
            'house_edge_percent': game_obj.calculate_house_edge(bet_amt)
        }
    
    if game_type == "Compare Both":
        results = {}
        for gtype in ["Fair Game", "Tweaked Game"]:
            if gtype == "Fair Game":
                game = EnhancedColorGame(gtype)
            else:
                game = EnhancedColorGame(gtype, tweak_method, house_edge)
            
            results[gtype] = simulate_game(game, num_simulations, bet_amount, 
                                          initial_balance, player_color)
        return results
    else:
        if game_type == "Tweaked Game":
            game = EnhancedColorGame(game_type, tweak_method, house_edge)
        else:
            game = EnhancedColorGame(game_type)
        
        return simulate_game(game, num_simulations, bet_amount, 
                            initial_balance, player_color)

def calculate_kelly_criterion(win_prob, win_multiplier, loss_multiplier=1):
    """Calculate Kelly Criterion for optimal bet sizing"""
    if win_prob <= 0 or win_multiplier <= 0:
        return 0
    b = win_multiplier / loss_multiplier - 1
    if b <= 0:
        return 0
    kelly = (win_prob * (b + 1) - 1) / b
    return max(0, min(kelly, 1))  # Clamp between 0 and 1

def calculate_risk_of_ruin(win_prob, win_amount, loss_amount, initial_balance, bet_amount):
    """Calculate risk of ruin using random walk approximation"""
    if win_prob <= 0 or win_prob >= 1:
        return 0 if win_prob >= 1 else 1
    
    p = win_prob
    q = 1 - p
    win_ratio = win_amount / bet_amount
    loss_ratio = loss_amount / bet_amount
    
    # Simplified risk of ruin calculation
    if p <= 0.5:
        return 1.0
    else:
        # Using classical gambler's ruin formula
        try:
            z = -loss_ratio / win_ratio
            risk = (z * (initial_balance / bet_amount) - 1) / (z * (initial_balance / bet_amount) - z ** (-initial_balance / bet_amount))
            return min(1, max(0, risk))
        except:
            return 0.5

def perform_statistical_tests(fair_results, tweaked_results):
    """Perform statistical tests between fair and tweaked results"""
    fair_payouts = np.array(fair_results['all_payouts'])
    tweaked_payouts = np.array(tweaked_results['all_payouts'])
    
    # T-test for means
    t_stat, p_value = stats.ttest_ind(fair_payouts, tweaked_payouts, equal_var=False)
    
    # Calculate confidence intervals
    fair_mean = np.mean(fair_payouts)
    tweaked_mean = np.mean(tweaked_payouts)
    fair_ci = stats.t.interval(0.95, len(fair_payouts)-1, loc=fair_mean, scale=stats.sem(fair_payouts))
    tweaked_ci = stats.t.interval(0.95, len(tweaked_payouts)-1, loc=tweaked_mean, scale=stats.sem(tweaked_payouts))
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p_value = stats.mannwhitneyu(fair_payouts, tweaked_payouts)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'fair_mean_ci': fair_ci,
        'tweaked_mean_ci': tweaked_ci,
        'u_statistic': u_stat,
        'u_p_value': u_p_value,
        'mean_difference': tweaked_mean - fair_mean,
        'effect_size': (tweaked_mean - fair_mean) / np.sqrt(
            (np.var(fair_payouts) + np.var(tweaked_payouts)) / 2
        )
    }

# Header with enhanced design
st.markdown("<h1 class='main-title'>🎲 FILIPINO PERYA COLOR GAME SIMULATOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'> Modeling & Simulation with Multiple House Edge Techniques | CSEC 413 Final Project</p>", unsafe_allow_html=True)

# Sidebar with improved design
with st.sidebar:
    st.markdown("### 🎮 CONTROL PANEL")
    
    # Game parameters in expandable sections
    with st.expander("⚙️ SIMULATION PARAMETERS", expanded=True):
        num_simulations = st.slider(
            "Number of Games", 
            1000, 100000, 10000, 1000,
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
                min_value=100.0, max_value=100000.0, value=1000.0, step=100.0,
                help="Starting amount of money"
            )
    
    with st.expander("🎯 GAME SETTINGS", expanded=True):
        game_type = st.radio(
            "Game Type:",
            ["Fair Game", "Tweaked Game", "Compare Both"],
            index=2,
            help="Fair: Equal probabilities | Tweaked: House advantage | Compare: Side-by-side analysis"
        )
        
        # Color selection
        colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
        player_color = st.selectbox(
            "Your Color:",
            colors,
            help="Choose which color to bet on"
        )
    
    # Advanced settings
    with st.expander("🔧 ADVANCED SETTINGS", expanded=False):
        if game_type == "Tweaked Game" or game_type == "Compare Both":
            house_edge = st.slider(
                "House Edge (%)", 
                0.1, 25.0, 5.0, 0.1,
                help="Casino's mathematical advantage over players"
            )
            
            tweak_method = st.selectbox(
                "Tweak Method:", 
                ["Weighted Probabilities", "Modified Payouts", "Normal Distribution", "Both"],
                help="How the house edge is implemented"
            )
            
            if tweak_method == "Modified Payouts" or tweak_method == "Both":
                payout_reduction = st.slider(
                    "Payout Reduction (%)",
                    0.0, 20.0, house_edge/2, 0.1,
                    help="How much to reduce the payout multiplier"
                )
            else:
                payout_reduction = 0.0
            
            if tweak_method == "Normal Distribution":
                col1, col2 = st.columns(2)
                with col1:
                    payout_std = st.slider(
                        "Payout Volatility (σ)",
                        1.0, 30.0, 15.0, 0.5,
                        help="Standard deviation of normal distribution payouts"
                    )
        
        # Risk management settings
        st.markdown("#### 📊 Risk Management")
        show_kelly = st.checkbox("Show Kelly Criterion", value=True)
        show_risk_of_ruin = st.checkbox("Show Risk of Ruin", value=True)
        
        # Statistical tests
        st.markdown("#### 🔬 Statistical Tests")
        perform_t_tests = st.checkbox("Perform Statistical Tests", value=True)
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95, 1)
    
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
            tweaked_ev = round(fair_ev * (1 - house_edge/100), 2)
            st.metric("Tweaked EV", f"₱{tweaked_ev}")
    
    # Information
    st.markdown("---")
    with st.expander("ℹ️ ABOUT ENHANCED FEATURES"):
        st.info("""
        *New Features:*
        
        1. *Multiple Tweak Methods:*
           - Weighted Probabilities
           - Modified Payouts  
           - Normal Distribution
           - Combined Tweaks
        
        2. *Advanced Statistics:*
           - Statistical hypothesis testing
           - Confidence intervals
           - Effect size calculations
        
        3. *Risk Management:*
           - Kelly Criterion
           - Risk of Ruin
           - Optimal bet sizing
        """)

# Main content
if run_simulation:
    # Progress animation
    with st.spinner('🎲 Configuring enhanced simulation...'):
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
        status_text.text(f"🎯 Running enhanced simulation... {i+1}% complete")
        
        # Update animation placeholder
        if i % 20 == 0:
            with simulation_placeholder.container():
                # Create a simple animation
                symbols = ['🎰', '🎲', '♠️', '♥️', '♦️', '♣️']
                current_symbol = symbols[i // 20 % 6]
                st.markdown(f"<h3 style='text-align: center; font-size: 4rem;'>{current_symbol}</h3>", unsafe_allow_html=True)
    
    status_text.success("✅ Enhanced simulation complete!")
    simulation_placeholder.empty()
    
    # Get simulation parameters
    tweak_params = {
        'tweak_method': tweak_method if game_type != "Fair Game" else "None",
        'house_edge': house_edge if game_type != "Fair Game" else 0,
        'payout_reduction': payout_reduction if 'payout_reduction' in locals() else 0
    }
    
    # Run the simulation
    if game_type == "Compare Both":
        results = run_monte_carlo_simulation(
            game_type, num_simulations, bet_amount, 
            initial_balance, player_color,
            tweak_method, house_edge
        )
        
        # Statistical tests
        if perform_t_tests:
            statistical_results = perform_statistical_tests(
                results["Fair Game"], results["Tweaked Game"]
            )
        
        # Comparison header
        st.markdown("<h2 class='section-header'>📊 COMPARATIVE ANALYSIS</h2>", unsafe_allow_html=True)
        
        # Tweak method description
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown(f"### 🎯 Tweak Method: *{tweak_method}*")
        if game_type != "Fair Game":
            st.markdown(f"*House Edge Target:* {house_edge}%")
            st.markdown(f"*Tweak Description:* {results['Tweaked Game']['tweak_description']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create two columns for comparison with badges
        col1, col2 = st.columns(2)
        
        for idx, (gtype, result) in enumerate(results.items()):
            with col1 if idx == 0 else col2:
                # Header with badge
                badge_class = "badge-fair" if gtype == "Fair Game" else "badge-tweaked"
                st.markdown(f"<h3><span class='badge {badge_class}'>{gtype.upper()}</span></h3>", unsafe_allow_html=True)
                
                # Tweak description for tweaked game
                if gtype == "Tweaked Game":
                    st.markdown(f"{result['tweak_description']}")
                
                # Metrics in cards
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
                    ev_icon = "📈" if result['expected_value'] > 0 else "📉"
                    st.metric("Expected Value", f"₱{result['expected_value']:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # House edge calculation
                if gtype == "Tweaked Game":
                    st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
                    st.markdown("#### 🏠 House Edge Analysis")
                    theoretical_edge = result['house_edge_percent']
                    empirical_edge = (results['Fair Game']['expected_value'] - result['expected_value']) / bet_amount * 100
                    st.metric("Theoretical Edge", f"{theoretical_edge:.2f}%")
                    st.metric("Empirical Edge", f"{empirical_edge:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Risk management metrics
                if show_kelly or show_risk_of_ruin:
                    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                    st.markdown("#### ⚖️ Risk Management")
                    
                    win_prob = result['win_rate']
                    win_mult = result['payout_multiplier']
                    
                    if show_kelly:
                        kelly = calculate_kelly_criterion(win_prob, win_mult)
                        optimal_bet = kelly * initial_balance
                        st.metric("Kelly Criterion", f"{kelly*100:.1f}%")
                        st.metric("Optimal Bet", f"₱{optimal_bet:.2f}")
                    
                    if show_risk_of_ruin:
                        risk = calculate_risk_of_ruin(
                            win_prob, 
                            bet_amount * win_mult, 
                            bet_amount,
                            initial_balance, 
                            bet_amount
                        )
                        st.metric("Risk of Ruin", f"{risk*100:.1f}%")
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
        
        # Statistical tests section
        if perform_t_tests:
            st.markdown("<h3 class='section-header'>🔬 STATISTICAL HYPOTHESIS TESTING</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("##### T-Test Results")
                st.markdown(f"*T-statistic:* {statistical_results['t_statistic']:.4f}")
                st.markdown(f"*P-value:* {statistical_results['p_value']:.6f}")
                
                if statistical_results['significant']:
                    st.success("✅ *Statistically Significant Difference* (p < 0.05)")
                    st.markdown("The house edge has a statistically significant impact on outcomes.")
                else:
                    st.warning("⚠️ *Not Statistically Significant* (p ≥ 0.05)")
                    st.markdown("The observed difference could be due to random chance.")
                
                st.markdown(f"*Effect Size (Cohen's d):* {statistical_results['effect_size']:.3f}")
                if abs(statistical_results['effect_size']) < 0.2:
                    st.markdown("Effect Size: Small")
                elif abs(statistical_results['effect_size']) < 0.5:
                    st.markdown("Effect Size: Medium")
                else:
                    st.markdown("Effect Size: Large")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("##### Confidence Intervals")
                st.markdown(f"*{confidence_level}% CI for Fair Game Mean:*")
                st.markdown(f"₱{statistical_results['fair_mean_ci'][0]:.3f} to ₱{statistical_results['fair_mean_ci'][1]:.3f}")
                
                st.markdown(f"*{confidence_level}% CI for Tweaked Game Mean:*")
                st.markdown(f"₱{statistical_results['tweaked_mean_ci'][0]:.3f} to ₱{statistical_results['tweaked_mean_ci'][1]:.3f}")
                
                st.markdown("##### Non-parametric Test")
                st.markdown(f"*Mann-Whitney U:* {statistical_results['u_statistic']:,.0f}")
                st.markdown(f"*U-test P-value:* {statistical_results['u_p_value']:.6f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualize the difference
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=results["Fair Game"]['all_payouts'],
                name="Fair Game",
                marker_color='#4CAF50',
                boxmean=True
            ))
            fig.add_trace(go.Box(
                y=results["Tweaked Game"]['all_payouts'],
                name="Tweaked Game",
                marker_color='#FF9800',
                boxmean=True
            ))
            fig.update_layout(
                title="Distribution of Payouts: Fair vs Tweaked",
                yaxis_title="Payout (₱)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("<h3 class='section-header'>📈 DETAILED COMPARISON</h3>", unsafe_allow_html=True)
        
        fair_result = results["Fair Game"]
        tweaked_result = results["Tweaked Game"]
        
        comparison_data = {
            "Metric": ["Final Balance", "Total Profit", "Win Rate", "Expected Value", 
                      "Theoretical EV", "House Edge (%)", "Total Wins", "Max Drawdown",
                      "Volatility (Std)", "Sharpe Ratio"],
            "Fair Game": [
                fair_result['final_balance'],
                fair_result['total_profit'],
                fair_result['win_rate'] * 100,
                fair_result['expected_value'],
                fair_result['theoretical_ev'],
                0.0,
                fair_result['wins'],
                (min(fair_result['history']) - initial_balance) / initial_balance * 100,
                np.std(fair_result['all_payouts']),
                fair_result['expected_value'] / np.std(fair_result['all_payouts']) if np.std(fair_result['all_payouts']) > 0 else 0
            ],
            "Tweaked Game": [
                tweaked_result['final_balance'],
                tweaked_result['total_profit'],
                tweaked_result['win_rate'] * 100,
                tweaked_result['expected_value'],
                tweaked_result['theoretical_ev'],
                tweaked_result['house_edge_percent'],
                tweaked_result['wins'],
                (min(tweaked_result['history']) - initial_balance) / initial_balance * 100,
                np.std(tweaked_result['all_payouts']),
                tweaked_result['expected_value'] / np.std(tweaked_result['all_payouts']) if np.std(tweaked_result['all_payouts']) > 0 else 0
            ],
            "Difference": [
                tweaked_result['final_balance'] - fair_result['final_balance'],
                tweaked_result['total_profit'] - fair_result['total_profit'],
                (tweaked_result['win_rate'] - fair_result['win_rate']) * 100,
                tweaked_result['expected_value'] - fair_result['expected_value'],
                tweaked_result['theoretical_ev'] - fair_result['theoretical_ev'],
                tweaked_result['house_edge_percent'],
                tweaked_result['wins'] - fair_result['wins'],
                (min(tweaked_result['history']) - min(fair_result['history'])) / initial_balance * 100,
                np.std(tweaked_result['all_payouts']) - np.std(fair_result['all_payouts']),
                (tweaked_result['expected_value'] / np.std(tweaked_result['all_payouts']) if np.std(tweaked_result['all_payouts']) > 0 else 0) -
                (fair_result['expected_value'] / np.std(fair_result['all_payouts']) if np.std(fair_result['all_payouts']) > 0 else 0)
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Style the comparison table
        def color_diff(val):
            if isinstance(val, (int, float)):
                if "Edge" in df_comparison.columns or "Drawdown" in df_comparison.columns:
                    # For these metrics, negative is bad for player
                    color = 'red' if val > 0 else 'green' if val < 0 else 'gray'
                else:
                    # For other metrics, positive is good
                    color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}; font-weight: bold;'
            return ''
        
        st.dataframe(
            df_comparison.style.format({
                "Fair Game": "{:,.2f}",
                "Tweaked Game": "{:,.2f}",
                "Difference": "{:+,.2f}"
            }).applymap(color_diff, subset=['Difference']),
            use_container_width=True,
            height=400
        )
        
        # Download option for comparison results
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### 📥 DOWNLOAD ENHANCED RESULTS")
        
        all_data = []
        for gtype, result in results.items():
            n = len(result['history'])
            for i in range(n):
                all_data.append({
                    'Game_Type': gtype,
                    'Game_Number': i + 1,
                    'Balance': result['history'][i] if i < len(result['history']) else None,
                    'Payout': result['all_payouts'][i] if i < len(result['all_payouts']) else None,
                    'Win': result['win_history'][i] if i < len(result['win_history']) else None,
                    'Winning_Color': result['winning_colors'][i] if i < len(result['winning_colors']) else None,
                    'Cumulative_Profit': (result['history'][i] - initial_balance) if i < len(result['history']) else None
                })
        
        if all_data:
            result_df = pd.DataFrame(all_data)
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📊 Download All Enhanced Simulation Data (CSV)",
                data=csv,
                file_name="enhanced_perya_game_comparison.csv",
                mime="text/csv",
                help="Download complete enhanced simulation data for further analysis"
            )
        
    else:
        # Single game type results
        result = run_monte_carlo_simulation(
            game_type, num_simulations, bet_amount, 
            initial_balance, player_color,
            tweak_method if game_type == "Tweaked Game" else "None",
            house_edge if game_type == "Tweaked Game" else 0
        )
        
        # Header with game type badge
        badge_type = "badge-fair" if game_type == "Fair Game" else "badge-tweaked"
        st.markdown(f"<h2 class='section-header'><span class='badge {badge_type}'>{game_type.upper()}</span> SIMULATION RESULTS</h2>", unsafe_allow_html=True)
        
        # Tweak method description
        if game_type == "Tweaked Game":
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown(f"### 🎯 Applied Tweak: *{tweak_method}*")
            st.markdown(f"*{result['tweak_description']}*")
            st.markdown(f"*Target House Edge:* {house_edge}%")
            st.markdown(f"*Calculated House Edge:* {result['house_edge_percent']:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
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
        
        # Risk management section
        if show_kelly or show_risk_of_ruin:
            st.markdown("### ⚖️ RISK MANAGEMENT ANALYSIS")
            
            col_risk1, col_risk2, col_risk3 = st.columns(3)
            
            with col_risk1:
                if show_kelly:
                    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                    st.markdown("#### 💰 Kelly Criterion")
                    win_prob = result['win_rate']
                    win_mult = result['payout_multiplier']
                    kelly = calculate_kelly_criterion(win_prob, win_mult)
                    optimal_bet = kelly * initial_balance
                    st.metric("Kelly Fraction", f"{kelly*100:.2f}%")
                    st.metric("Optimal Bet Size", f"₱{optimal_bet:.2f}")
                    
                    if kelly > 0:
                        st.success(f"✅ Recommended bet: {kelly*100:.1f}% of bankroll (₱{optimal_bet:.2f})")
                    else:
                        st.warning("⚠️ Negative Kelly: Game is unfavorable, consider not betting")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col_risk2:
                if show_risk_of_ruin:
                    st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
                    st.markdown("#### ☠️ Risk of Ruin")
                    risk = calculate_risk_of_ruin(
                        result['win_rate'],
                        bet_amount * result['payout_multiplier'],
                        bet_amount,
                        initial_balance,
                        bet_amount
                    )
                    st.metric("Probability of Ruin", f"{risk*100:.2f}%")
                    
                    if risk < 0.01:
                        st.success("✅ Very low risk of ruin")
                    elif risk < 0.05:
                        st.info("ℹ️ Moderate risk of ruin")
                    elif risk < 0.20:
                        st.warning("⚠️ High risk of ruin")
                    else:
                        st.error("🚨 Very high risk of ruin")
                    
                    # Calculate safe bankroll
                    safe_bankroll = bet_amount * np.log(0.01) / np.log(1 - result['win_rate']) if result['win_rate'] < 1 else initial_balance
                    st.metric("Safe Bankroll", f"₱{safe_bankroll:.0f}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col_risk3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("#### 📊 Risk Metrics")
                
                volatility = np.std(result['all_payouts'])
                sharpe = result['expected_value'] / volatility if volatility > 0 else 0
                max_drawdown = (min(result['history']) - initial_balance) / initial_balance * 100
                profit_factor = abs(result['wins'] * bet_amount * result['payout_multiplier']) / abs((result['total_games'] - result['wins']) * bet_amount) if result['total_games'] > result['wins'] else float('inf')
                
                st.metric("Volatility (σ)", f"₱{volatility:.2f}")
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
                st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Balance History", 
            "🎯 Win Analysis", 
            "⚖️ Probability Analysis", 
            "📈 Advanced Statistics",
            "🔬 Payout Distribution"
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
            
            # Theoretical EV line
            if game_type == "Tweaked Game":
                theoretical_path = [initial_balance + result['theoretical_ev'] * i for i in range(len(result['history']))]
                fig.add_trace(
                    go.Scatter(
                        y=theoretical_path,
                        mode='lines',
                        name='Theoretical EV',
                        line=dict(color='#4CAF50', width=2, dash='dot')
                    ),
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
                        line=dict(color='#FF4B4B', width=2)
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
                    st.markdown(f"*Player's Color:* {player_color}")
                    st.markdown(f"*Bet Amount:* ₱{bet_amount:,.2f}")
                    if game_type == "Tweaked Game":
                        st.markdown(f"*Tweak Method:* {tweak_method}")
                        st.markdown(f"*House Edge:* {result['house_edge_percent']:.2f}%")
                
                with col2:
                    if game_type == "Tweaked Game":
                        st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
                        st.markdown("#### ⚠️ House Edge Analysis")
                        fair_ev = (1/6) * bet_amount * 5 - (5/6) * bet_amount
                        tweaked_ev = result['expected_value']
                        theoretical_ev = result['theoretical_ev']
                        house_edge_empirical = (fair_ev - tweaked_ev) / bet_amount * 100
                        st.markdown(f"*Theoretical House Edge:* {result['house_edge_percent']:.2f}%")
                        st.markdown(f"*Empirical House Edge:* {house_edge_empirical:.2f}%")
                        st.markdown(f"*Theoretical EV:* ₱{theoretical_ev:.3f} per game")
                        st.markdown(f"*Empirical EV:* ₱{tweaked_ev:.3f} per game")
                        st.markdown(f"*Fair EV:* ₱{fair_ev:.3f} per game")
        
        with tab4:
            # Statistical summary with enhanced metrics
            st.markdown("### 📊 Advanced Statistical Summary")
            
            history_series = pd.Series(result['history'])
            payout_series = pd.Series(result['all_payouts'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("##### Balance Statistics")
                
                # Calculate various statistical measures
                stats_data = {
                    'Statistic': ['Mean Balance', 'Median Balance', 'Std Deviation', 
                                'Skewness', 'Kurtosis', 'Minimum Balance', 
                                'Maximum Balance', 'Range', 'IQR',
                                'Coefficient of Variation'],
                    'Value': [
                        history_series.mean(),
                        history_series.median(),
                        history_series.std(),
                        history_series.skew(),
                        history_series.kurtosis(),
                        history_series.min(),
                        history_series.max(),
                        history_series.max() - history_series.min(),
                        history_series.quantile(0.75) - history_series.quantile(0.25),
                        (history_series.std() / history_series.mean()) * 100 if history_series.mean() != 0 else 0
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(
                    stats_df.style.format({'Value': '{:,.4f}'}),
                    use_container_width=True,
                    height=400
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("##### Payout Statistics")
                
                payout_stats = {
                    'Statistic': ['Mean Payout', 'Median Payout', 'Std Payout',
                                'Payout Skewness', 'Payout Kurtosis', 'Min Payout',
                                'Max Payout', 'Positive Payouts', 'Negative Payouts',
                                'Zero Payouts'],
                    'Value': [
                        payout_series.mean(),
                        payout_series.median(),
                        payout_series.std(),
                        payout_series.skew(),
                        payout_series.kurtosis(),
                        payout_series.min(),
                        payout_series.max(),
                        (payout_series > 0).sum(),
                        (payout_series < 0).sum(),
                        (payout_series == 0).sum()
                    ]
                }
                payout_stats_df = pd.DataFrame(payout_stats)
                st.dataframe(
                    payout_stats_df.style.format({'Value': '{:,.4f}'}),
                    use_container_width=True,
                    height=400
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Normality test
            st.markdown("##### 🧪 Normality Tests")
            col_norm1, col_norm2 = st.columns(2)
            
            with col_norm1:
                st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                st.markdown("###### Shapiro-Wilk Test")
                if len(payout_series) <= 5000:  # Shapiro-Wilk has limit
                    shapiro_stat, shapiro_p = stats.shapiro(payout_series)
                    st.metric("W-statistic", f"{shapiro_stat:.4f}")
                    st.metric("P-value", f"{shapiro_p:.6f}")
                    if shapiro_p < 0.05:
                        st.warning("❌ Reject normality: Payouts are not normally distributed")
                    else:
                        st.success("✅ Cannot reject normality")
                else:
                    st.info("Dataset too large for Shapiro-Wilk test")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_norm2:
                st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                st.markdown("###### Kolmogorov-Smirnov Test")
                # Test against normal distribution with same mean and std
                ks_stat, ks_p = stats.kstest(payout_series, 'norm', 
                                           args=(payout_series.mean(), payout_series.std()))
                st.metric("KS-statistic", f"{ks_stat:.4f}")
                st.metric("P-value", f"{ks_p:.6f}")
                if ks_p < 0.05:
                    st.warning("❌ Reject normality")
                else:
                    st.success("✅ Cannot reject normality")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # QQ Plot
            st.markdown("##### 📊 Q-Q Plot (Normality Check)")
            fig = go.Figure()
            
            # Theoretical quantiles
            theoretical_q = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(payout_series))))
            actual_q = np.sort(payout_series)
            
            fig.add_trace(go.Scatter(
                x=theoretical_q,
                y=actual_q,
                mode='markers',
                name='Q-Q Points',
                marker=dict(size=6, color='#2196F3')
            ))
            
            # Add reference line
            min_val = min(theoretical_q.min(), actual_q.min())
            max_val = max(theoretical_q.max(), actual_q.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Q-Q Plot: Checking Normality of Payouts",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            # Payout distribution analysis
            st.markdown("### 💰 Payout Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of payouts
                fig = px.histogram(
                    result['all_payouts'],
                    nbins=50,
                    title="Distribution of Payouts",
                    labels={'value': 'Payout (₱)', 'count': 'Frequency'},
                    color_discrete_sequence=['#2196F3']
                )
                fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Break-even"
                )
                fig.add_vline(
                    x=np.mean(result['all_payouts']),
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Mean: ₱{np.mean(result['all_payouts']):.2f}"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cumulative distribution function
                sorted_payouts = np.sort(result['all_payouts'])
                cdf = np.arange(1, len(sorted_payouts) + 1) / len(sorted_payouts)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sorted_payouts,
                    y=cdf,
                    mode='lines',
                    name='CDF',
                    line=dict(color='#FF4B4B', width=2)
                ))
                
                # Add vertical lines for key percentiles
                percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
                for p in percentiles:
                    percentile_val = np.percentile(result['all_payouts'], p * 100)
                    fig.add_vline(
                        x=percentile_val,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"{p*100:.0f}%: ₱{percentile_val:.2f}"
                    )
                
                fig.update_layout(
                    title="Cumulative Distribution Function (CDF)",
                    xaxis_title="Payout (₱)",
                    yaxis_title="Cumulative Probability",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Payout statistics table
            st.markdown("##### 📋 Payout Percentiles")
            percentile_data = {
                'Percentile': ['5th', '25th (Q1)', '50th (Median)', '75th (Q3)', '95th', '99th'],
                'Value (₱)': [
                    np.percentile(result['all_payouts'], 5),
                    np.percentile(result['all_payouts'], 25),
                    np.percentile(result['all_payouts'], 50),
                    np.percentile(result['all_payouts'], 75),
                    np.percentile(result['all_payouts'], 95),
                    np.percentile(result['all_payouts'], 99)
                ],
                'Interpretation': [
                    "5% of payouts are below this",
                    "25% of payouts are below this",
                    "Median payout",
                    "75% of payouts are below this",
                    "95% of payouts are below this",
                    "99% of payouts are below this"
                ]
            }
            
            percentile_df = pd.DataFrame(percentile_data)
            st.dataframe(
                percentile_df.style.format({'Value (₱)': '₱{:,.2f}'}),
                use_container_width=True
            )
    
    # Download results for single game
    if game_type != "Compare Both":
        actual_games = len(result['history'])
        
        # Create enhanced result DataFrame
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
        
        payouts = result.get('all_payouts', [0] * actual_games)
        if len(payouts) < actual_games:
            payouts.extend([0] * (actual_games - len(payouts)))
        
        result_df = pd.DataFrame({
            'Game_Number': game_numbers[:actual_games],
            'Balance': balances[:actual_games],
            'Payout': payouts[:actual_games],
            'Win': win_history[:actual_games],
            'Winning_Color': winning_colors[:actual_games],
            'Cumulative_Profit': [bal - initial_balance for bal in balances[:actual_games]]
        })
        
        csv = result_df.to_csv(index=False)
        
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("### 📥 DOWNLOAD ENHANCED SIMULATION DATA")
        st.download_button(
            label=f"📊 Download {game_type} Enhanced Data (CSV)",
            data=csv,
            file_name=f"enhanced_perya_{game_type.replace(' ', '_').lower()}_simulation.csv",
            mime="text/csv",
            help="Download complete enhanced simulation data for further analysis"
        )
    
    # Simulation insights with enhanced presentation
    st.markdown("<h2 class='section-header'>🧠 SIMULATION INSIGHTS</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Mathematical Insights")
        st.markdown("""
        *Multiple House Edge Methods:*
        - Weighted probabilities shift win likelihood
        - Modified payouts reduce win amounts
        - Normal distribution creates variance
        - Combined methods maximize edge
        
        *Statistical Significance:*
        - Hypothesis tests validate results
        - Confidence intervals show uncertainty
        - Effect sizes measure impact magnitude
        """)
    
    with col2:
        st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚠️ Risk Management Insights")
        st.markdown("""
        *Kelly Criterion:*
        - Optimal bet sizing formula
        - Maximizes long-term growth
        - Prevents over-betting
        
        *Risk of Ruin:*
        - Probability of losing everything
        - Depends on edge and bankroll
        - Key for survival strategy
        
        *Bankroll Management:*
        - Bet size relative to edge
        - Volatility considerations
        - Drawdown limits
        """)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Advanced Analysis")
        st.markdown("""
        *Distribution Analysis:*
        - Normality testing
        - Percentile analysis
        - Skewness and kurtosis
        
        *Statistical Tests:*
        - T-tests for means
        - Non-parametric tests
        - Confidence intervals
        
        *Practical Applications:*
        - Game design optimization
        - Risk assessment
        - Profitability analysis
        """)

else:
    # Default view - Landing page with enhanced design
    st.markdown("<h2 style='color: #2E4053; margin-bottom: 2rem; text-align: center;'>🎯 PROJECT OVERVIEW</h2>", unsafe_allow_html=True)
    
    # Introduction cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🎮 Enhanced Game")
        st.markdown("""
        Traditional Filipino "Perya" with:
        - 4 house edge methods
        - Statistical testing
        - Risk management tools
        - Advanced analytics
        """)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Advanced Science")
        st.markdown("""
        Enhanced Monte Carlo:
        - Multiple tweak methods
        - Hypothesis testing
        - Confidence intervals
        - Distribution analysis
        """)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Enhanced Purpose")
        st.markdown("""
        Advanced Education:
        - Multiple edge strategies
        - Statistical validation
        - Risk quantification
        - Professional analysis
        """)
    
    # New features showcase
    st.markdown("<h3 class='section-header'>🚀  FEATURES</h3>", unsafe_allow_html=True)
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        ### 📋 New Tweak Methods
        
        1. *Weighted Probabilities*
           - Adjust win probabilities
           - Shift likelihood to house colors
           - Maintains apparent fairness
        
        2. *Modified Payouts*
           - Reduce win amounts
           - Keep probabilities fair
           - Hidden edge in payouts
        
        3. *Normal Distribution*
           - Variable payout amounts
           - Negative mean for edge
           - High variance for excitement
        
        4. *Combined Methods*
           - Multiple edge sources
           - Smaller individual tweaks
           - Harder to detect
        """)
    
    with features_col2:
        st.markdown("""
        ### 🔬 Advanced Analytics
        
        *Statistical Testing:*
        - T-tests for significance
        - Confidence intervals
        - Effect size calculation
        - Non-parametric tests
        
        *Risk Management:*
        - Kelly Criterion
        - Risk of Ruin
        - Optimal bet sizing
        - Bankroll safety
        
        *Distribution Analysis:*
        - Normality tests
        - Percentile analysis
        - Skewness & kurtosis
        - Q-Q plots
        """)
    
    # Tweak method visualization
    st.markdown("<h3 class='section-header'>🎯 HOUSE EDGE METHODS VISUALIZATION</h3>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Weighted Probabilities", "Modified Payouts", "Normal Distribution", "Combined"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            fair_probs = [1/6] * 6
            tweaked_probs = [0.12, 0.12, 0.12, 0.213, 0.213, 0.214]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black'],
                y=fair_probs,
                name='Fair',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black'],
                y=tweaked_probs,
                name='Weighted',
                marker_color='coral'
            ))
            fig.update_layout(
                title="Weighted Probabilities Method",
                yaxis_title="Probability",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            #### How it works:
            
            *Player colors (Red, Green, Blue):*
            - Probability reduced by ~29%
            - From 16.67% to 12.0%
            
            *House colors (Yellow, White, Black):*
            - Probability increased by ~28%
            - From 16.67% to 21.33%
            
            *Result:*
            - Player win probability: 36% vs 50% fair
            - House edge: ~14%
            - Payout remains 5:1
            """)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            multipliers = ['5.0', '4.75', '4.5', '4.25', '4.0']
            house_edges = ['0%', '5%', '10%', '15%', '20%']
            
            fig = go.Figure(data=[go.Bar(
                x=house_edges,
                y=multipliers,
                orientation='h',
                marker_color=['#4CAF50', '#FF9800', '#FF5722', '#F44336', '#D32F2F']
            )])
            fig.update_layout(
                title="Modified Payouts Method",
                xaxis_title="House Edge",
                yaxis_title="Payout Multiplier",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            #### How it works:
            
            *Fair Game:*
            - Payout: 5.0× bet
            - House edge: 0%
            
            *Tweaked Games:*
            - Payout: 4.75× (5% edge)
            - Payout: 4.5× (10% edge)
            - Payout: 4.25× (15% edge)
            - Payout: 4.0× (20% edge)
            
            *Advantages:*
            - Probabilities remain fair
            - Easy to calculate edge
            - Players may not notice
            """)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            x = np.linspace(-50, 100, 1000)
            fair_pdf = stats.norm.pdf(x, 0, 20)
            tweaked_pdf = stats.norm.pdf(x, -5, 25)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=fair_pdf,
                mode='lines',
                name='Fair (μ=0)',
                line=dict(color='#4CAF50', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=x, y=tweaked_pdf,
                mode='lines',
                name='Tweaked (μ=-5)',
                line=dict(color='#FF9800', width=3)
            ))
            fig.update_layout(
                title="Normal Distribution Payouts",
                xaxis_title="Payout (₱)",
                yaxis_title="Probability Density",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            #### How it works:
            
            *Fair Distribution:*
            - Mean payout: ₱0
            - High variance
            - Symmetrical
            
            *Tweaked Distribution:*
            - Mean payout: -₱5
            - Higher variance
            - Shifted left
            
            *Key Features:*
            - Negative mean creates edge
            - High variance = excitement
            - Occasional big wins
            - Overall loss guaranteed
            """)
    
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            methods = ['Probability\nTweak', 'Payout\nTweak', 'Combined\nTweak']
            edges = [7.0, 7.0, 14.0]
            
            fig = go.Figure(data=[go.Bar(
                x=methods,
                y=edges,
                marker_color=['#FF9800', '#2196F3', '#FF4B4B']
            )])
            fig.update_layout(
                title="Combined Method Edge Distribution",
                yaxis_title="House Edge (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            #### How it works:
            
            *Split Edge Strategy:*
            - Half from probabilities
            - Half from payouts
            - Total edge: sum of both
            
            *Example (14% edge):*
            - Probabilities: 7% edge
            - Payouts: 7% edge
            - Total: 14% edge
            
            *Advantages:*
            - Each tweak smaller
            - Harder to detect
            - More robust
            - Can adjust balance
            """)
    
    # Learning outcomes
    st.markdown("<h3 class='section-header'>🎓 LEARNING OUTCOMES</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    #### 📊 What You'll Learn:
    
    1. *Advanced Probability Theory*: Multiple methods to create house edge
    2. *Statistical Hypothesis Testing*: Validating results with statistical tests
    3. *Risk Management Mathematics*: Kelly Criterion and Risk of Ruin calculations
    4. *Distribution Analysis*: Understanding payout distributions and normality
    5. *Game Design Principles*: Balancing excitement with profitability
    6. *Professional Analytics*: Confidence intervals, effect sizes, and significance
    
    #### 🔍 Advanced Concepts Covered:
    
    - *Multiple Edge Creation*: Four different methods to establish house advantage
    - *Statistical Validation*: T-tests, confidence intervals, and p-values
    - *Risk Quantification*: Mathematical risk management tools
    - *Distribution Properties*: Skewness, kurtosis, and normality testing
    - *Optimal Strategies*: Kelly betting and bankroll management
    
    #### 💡 Professional Applications:
    
    - Casino game design and regulation
    - Financial risk modeling and assessment
    - Insurance product pricing
    - Statistical quality control in gaming
    - Academic research in probability
    - Regulatory compliance analysis
    """)

# Enhanced Footer
st.markdown("""
<div class='footer'>
    <h3>🎓 CSEC 413 -  Modeling and Simulation</h3>
    <p><strong>Final Project: Stochastic Game Simulation with Multiple House Edge Methods</strong></p>
    <p>This educational tool demonstrates advanced mathematical concepts behind casino games including statistical testing and risk management.</p>
    <p style='color: #FF4B4B; font-weight: bold;'>⚠️ Gambling involves significant risk of financial loss. This simulation is for educational purposes only.</p>
    <p>© 2024 Enhanced Filipino Perya Game Simulation | Made with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)