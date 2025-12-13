import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Perya Game Simulation",
    page_icon="🎰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0E1117;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🎲 Filipino Perya: Color Game Simulation</h1>", unsafe_allow_html=True)
st.markdown("### Modeling and Simulation of Casino Games with House Edge Analysis")

# Sidebar for controls
with st.sidebar:
    st.header("🎮 Game Controls")
    
    # Game parameters
    st.subheader("Simulation Parameters")
    num_simulations = st.slider("Number of Games per Simulation", 1000, 50000, 10000, 1000)
    bet_amount = st.number_input("Bet Amount per Game (PHP)", min_value=1.0, max_value=1000.0, value=10.0, step=10.0)
    initial_balance = st.number_input("Player Initial Balance (PHP)", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
    
    # Game type selection
    game_type = st.radio("Select Game Type:", ["Fair Game", "Tweaked Game", "Compare Both"])
    
    # Color selection
    st.subheader("Betting Strategy")
    colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
    player_color = st.selectbox("Choose your color:", colors)
    
    # Run simulation button
    run_simulation = st.button("🚀 Run Simulation", type="primary")
    
    # Additional controls
    st.subheader("Advanced Settings")
    show_advanced = st.checkbox("Show Advanced Settings")
    
    if show_advanced:
        if game_type == "Tweaked Game" or game_type == "Compare Both":
            house_edge = st.slider("House Edge (%)", 1.0, 20.0, 5.0, 0.5)
            tweak_type = st.selectbox("Tweak Method:", 
                                     ["Weighted Probabilities", "Modified Payouts", "Both"])

# Game logic functions
class ColorGame:
    def __init__(self, game_type="Fair Game", house_edge=5.0):
        self.colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
        self.game_type = game_type
        self.house_edge = house_edge / 100  # Convert to decimal
        
        if game_type == "Fair Game":
            self.setup_fair_game()
        else:
            self.setup_tweaked_game()
    
    def setup_fair_game(self):
        # Fair probabilities (1/6 each)
        self.probabilities = [1/6] * 6
        self.payout_multiplier = 5.0  # 5-to-1 payout for fair game
    
    def setup_tweaked_game(self):
        # Create house edge through weighted probabilities
        base_prob = 1/6
        reduction = self.house_edge / 6
        
        # Reduce probability for player-favored colors, increase for others
        self.probabilities = []
        for i in range(6):
            if i < 3:  # First three colors have lower probability
                prob = base_prob - reduction
            else:  # Last three colors have higher probability
                prob = base_prob + reduction
            self.probabilities.append(prob)
        
        # Normalize probabilities
        total = sum(self.probabilities)
        self.probabilities = [p/total for p in self.probabilities]
        
        # Slightly reduce payout
        self.payout_multiplier = 4.8  # Reduced from 5.0
    
    def play_round(self, player_color, bet_amount):
        # Determine winning color based on probabilities
        winning_color = np.random.choice(self.colors, p=self.probabilities)
        
        # Check if player wins
        if player_color == winning_color:
            win_amount = bet_amount * self.payout_multiplier
            return True, win_amount, winning_color
        else:
            return False, -bet_amount, winning_color

def run_monte_carlo_simulation(game_type, num_simulations, bet_amount, initial_balance, player_color):
    """Run Monte Carlo simulation"""
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
                    # Player is bankrupt
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
        # Single game type simulation
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

# Main app logic
if run_simulation:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run simulation
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        status_text.text(f"Running simulation... {i+1}%")
    
    status_text.text("Simulation complete!")
    
    # Run the simulation
    if game_type == "Compare Both":
        results = run_monte_carlo_simulation(
            game_type, num_simulations, bet_amount, 
            initial_balance, player_color
        )
        
        # Create two columns for comparison
        col1, col2 = st.columns(2)
        
        for idx, (gtype, result) in enumerate(results.items()):
            with col1 if idx == 0 else col2:
                st.markdown(f"### {gtype}")
                
                # Metrics
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("Final Balance", f"₱{result['final_balance']:,.2f}")
                    st.metric("Total Profit", f"₱{result['total_profit']:,.2f}", 
                             delta=f"{result['total_profit']/initial_balance*100:.1f}%")
                with col_metric2:
                    st.metric("Win Rate", f"{result['win_rate']*100:.1f}%")
                    st.metric("Expected Value", f"₱{result['expected_value']:.2f}")
                
                # Plot balance history
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=result['history'],
                    mode='lines',
                    name='Balance',
                    line=dict(color='green' if result['final_balance'] >= initial_balance else 'red')
                ))
                fig.update_layout(
                    title=f"Balance Over Time - {gtype}",
                    xaxis_title="Game Number",
                    yaxis_title="Balance (PHP)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Comparison metrics
        st.markdown("### 📊 Comparative Analysis")
        
        fair_result = results["Fair Game"]
        tweaked_result = results["Tweaked Game"]
        
        comparison_data = {
            "Metric": ["Final Balance", "Total Profit", "Win Rate", "Expected Value"],
            "Fair Game": [
                fair_result['final_balance'],
                fair_result['total_profit'],
                fair_result['win_rate'] * 100,
                fair_result['expected_value']
            ],
            "Tweaked Game": [
                tweaked_result['final_balance'],
                tweaked_result['total_profit'],
                tweaked_result['win_rate'] * 100,
                tweaked_result['expected_value']
            ],
            "Difference": [
                tweaked_result['final_balance'] - fair_result['final_balance'],
                tweaked_result['total_profit'] - fair_result['total_profit'],
                (tweaked_result['win_rate'] - fair_result['win_rate']) * 100,
                tweaked_result['expected_value'] - fair_result['expected_value']
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison.style.format({
            "Fair Game": "{:,.2f}",
            "Tweaked Game": "{:,.2f}",
            "Difference": "{:+,.2f}"
        }))
        
        # Download option for comparison results
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        # Create a combined DataFrame for download
        all_data = []
        for gtype, result in results.items():
            # Ensure we have the right number of entries
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
                label="📥 Download All Simulation Data",
                data=csv,
                file_name="color_game_comparison_simulation.csv",
                mime="text/csv"
            )
        
    else:
        # Single game type results
        result = run_monte_carlo_simulation(
            game_type, num_simulations, bet_amount, 
            initial_balance, player_color
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Balance", f"₱{result['final_balance']:,.2f}",
                     delta=f"{result['total_profit']:,.2f}")
        with col2:
            st.metric("Win Rate", f"{result['win_rate']*100:.2f}%",
                     delta=f"{result['wins']} wins")
        with col3:
            st.metric("Total Games", f"{result['total_games']:,}")
        with col4:
            ev_color = "green" if result['expected_value'] > 0 else "red"
            st.metric("Expected Value", f"₱{result['expected_value']:.3f}")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Balance History", "Win Distribution", 
                                         "Probability Analysis", "Statistical Summary"])
        
        with tab1:
            # Plot balance over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=result['history'],
                mode='lines',
                name='Player Balance',
                line=dict(color='blue')
            ))
            fig.add_hline(y=initial_balance, line_dash="dash", 
                         line_color="red", annotation_text="Initial Balance")
            fig.update_layout(
                title="Player Balance Over Time",
                xaxis_title="Game Number",
                yaxis_title="Balance (PHP)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Rolling average
            window_size = min(100, num_simulations // 10)
            if window_size > 1:
                rolling_avg = pd.Series(result['history']).rolling(window=window_size).mean()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=rolling_avg,
                    mode='lines',
                    name=f'{window_size}-Game Moving Average',
                    line=dict(color='orange', width=2)
                ))
                fig2.update_layout(
                    title=f"Moving Average Balance (Window: {window_size} games)",
                    xaxis_title="Game Number",
                    yaxis_title="Balance (PHP)",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            # Win/loss distribution
            col1, col2 = st.columns(2)
            with col1:
                win_loss_counts = pd.Series(result['win_history']).value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Losses', 'Wins'],
                    values=[win_loss_counts.get(False, 0), win_loss_counts.get(True, 0)],
                    hole=.3,
                    marker_colors=['red', 'green']
                )])
                fig.update_layout(title="Win/Loss Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Winning colors distribution
                if 'winning_colors' in result:
                    color_counts = pd.Series(result['winning_colors']).value_counts()
                    fig = px.bar(
                        x=color_counts.index,
                        y=color_counts.values,
                        title="Winning Colors Distribution",
                        labels={'x': 'Color', 'y': 'Frequency'},
                        color=color_counts.index
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Probability analysis
            if 'probabilities' in result:
                colors = ["Red", "Green", "Blue", "Yellow", "White", "Black"]
                prob_df = pd.DataFrame({
                    'Color': colors,
                    'Probability': result['probabilities'],
                    'Fair Probability': [1/6] * 6
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=prob_df['Color'],
                    y=prob_df['Fair Probability'],
                    name='Fair Probability',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=prob_df['Color'],
                    y=prob_df['Probability'],
                    name=f'{game_type} Probability',
                    marker_color='coral'
                ))
                fig.update_layout(
                    title="Probability Distribution Comparison",
                    barmode='group',
                    yaxis_title="Probability",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**Payout Multiplier:** {result['payout_multiplier']}:1")
                if game_type == "Tweaked Game":
                    fair_ev = (1/6) * bet_amount * 5 - (5/6) * bet_amount
                    tweaked_ev = result['expected_value']
                    house_edge = (fair_ev - tweaked_ev) / bet_amount * 100
                    st.markdown(f"**Calculated House Edge:** {house_edge:.2f}%")
        
        with tab4:
            # Statistical summary
            st.subheader("Statistical Analysis")
            
            history_series = pd.Series(result['history'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Balance Statistics")
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                    'Value': [
                        history_series.mean(),
                        history_series.median(),
                        history_series.std(),
                        history_series.min(),
                        history_series.max(),
                        history_series.skew(),
                        history_series.kurtosis()
                    ]
                })
                st.dataframe(stats_df.style.format({'Value': '{:,.2f}'}))
            
            with col2:
                st.markdown("##### Performance Metrics")
                # Create a DataFrame with mixed data types
                metrics_data = {
                    'Metric': ['Total Return', 'Return %', 'Sharpe Ratio*', 
                              'Max Drawdown', 'Volatility', 'Risk of Ruin*'],
                    'Value': [
                        result['total_profit'],
                        result['total_profit'] / initial_balance * 100,
                        result['total_profit'] / history_series.std() if history_series.std() > 0 else 0,
                        (history_series.min() - initial_balance) / initial_balance * 100,
                        history_series.std(),
                        "High" if result['final_balance'] < bet_amount else "Low"
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                
                # FIX: Use a function to format only numeric values
                def format_metrics(val):
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        return f"{val:,.2f}"
                    return val
                
                # Apply formatting to each value in the 'Value' column
                formatted_values = []
                for val in metrics_df['Value']:
                    formatted_values.append(format_metrics(val))
                
                metrics_df['Value'] = formatted_values
                st.dataframe(metrics_df)
                st.caption("*Approximate calculations")
    
    # Add simulation insights
    st.markdown("---")
    st.markdown("### 📈 Simulation Insights")
    
    insights = """
    #### Key Observations:
    1. **Law of Large Numbers**: As the number of games increases, the results converge to the expected value
    2. **House Edge Impact**: Even small probability tweaks significantly affect long-term outcomes
    3. **Volatility**: Short-term results can vary widely due to randomness
    4. **Bankroll Management**: Initial balance and bet size greatly affect survival probability
    
    #### Mathematical Foundation:
    - **Expected Value (EV)**: Average outcome per game
    - **House Edge**: Percentage advantage the casino has over players
    - **Monte Carlo Simulation**: Uses random sampling to model probabilistic systems
    """
    
    st.markdown(insights)
    
    # Download results - FIXED COMPLETELY
    if game_type != "Compare Both":
        # Get the actual number of games simulated (might be less than num_simulations if player went bankrupt early)
        actual_games = len(result['history'])
        
        # Create lists with exactly the same length
        game_numbers = list(range(1, actual_games + 1))
        balances = result['history']
        
        # Get win_history and winning_colors with proper length
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
        
        # Now create the DataFrame with all arrays having the same length
        result_df = pd.DataFrame({
            'Game_Number': game_numbers[:actual_games],
            'Balance': balances[:actual_games],
            'Win': win_history[:actual_games],
            'Winning_Color': winning_colors[:actual_games]
        })
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Simulation Data",
            data=csv,
            file_name=f"{game_type.replace(' ', '_').lower()}_simulation.csv",
            mime="text/csv"
        )

else:
    # Default view when simulation hasn't been run
    st.markdown("""
    ## 🎯 Project Overview
    
    This application simulates a Filipino "Perya" Color Game to demonstrate:
    
    ### 🎮 The Game Rules:
    1. Player bets on one of six colors
    2. A winning color is randomly selected
    3. If player's color wins, they get a payout
    4. Otherwise, they lose their bet
    
    ### 🔬 Simulation Objectives:
    1. **Model a Fair Game**: Equal probabilities (1/6 each) with fair payouts
    2. **Model a Tweaked Game**: Introduces house edge through probability weighting
    3. **Compare Results**: Analyze how small changes affect long-term outcomes
    
    ### 📊 Analysis Includes:
    - Monte Carlo simulation (10,000+ games)
    - Balance progression over time
    - Win/loss distribution
    - Expected value calculation
    - House edge quantification
    - Statistical analysis
    
    ### 🎲 How to Use:
    1. Adjust simulation parameters in the sidebar
    2. Select game type (Fair, Tweaked, or Compare Both)
    3. Choose your betting color
    4. Click "Run Simulation" to see results
    
    ### 🎯 Learning Outcomes:
    - Understand probability modeling
    - Analyze house edge impact
    - Visualize stochastic processes
    - Apply Monte Carlo methods
    """)
    
    # Add some sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Example: Fair Game Probabilities")
        fair_probs = [1/6] * 6
        fig = go.Figure(data=[go.Pie(
            labels=['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black'],
            values=fair_probs,
            hole=.3
        )])
        fig.update_layout(title="Equal Probabilities (16.67% each)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Example: Tweaked Game Probabilities")
        tweaked_probs = [0.14, 0.14, 0.14, 0.19, 0.19, 0.20]
        fig = go.Figure(data=[go.Pie(
            labels=['Red', 'Green', 'Blue', 'Yellow', 'White', 'Black'],
            values=tweaked_probs,
            hole=.3
        )])
        fig.update_layout(title="Weighted Probabilities (House Edge: ~5%)")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**CSEC 413 - Modeling and Simulation** | *Final Project: Stochastic Game Simulation*  
*This simulation demonstrates how casinos maintain profitability through mathematical edge*
""")