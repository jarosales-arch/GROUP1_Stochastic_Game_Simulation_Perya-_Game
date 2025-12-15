# ⬛⬜🟥 Stochastic Game Simulation: Filipino Perya Color Game 🟨🟦🟩

A comprehensive **Streamlit application** for modeling and simulating a traditional Filipino carnival game. This project demonstrates **probabilistic systems**, **Monte Carlo simulations**, and **house edge analysis** through interactive experimentation and visualization.

---

## 🎯 Features

### **1. Dual Game Models**
- **Fair Game**
  - Equal probabilities (16.67% per color)
  - 5-to-1 payout
- **Tweaked Game**
  - Weighted probabilities
  - Reduced payouts
  - Creates house edge
- **Comparison Mode**
  - Side-by-side analysis of both models

---

### **2. Interactive Simulation Controls**

#### **Game Parameters**
- Simulation size: **1,000 – 50,000** games  
- Bet amount: **PHP 1 – 1,000**  
- Initial balance: **PHP 100 – 10,000**

#### **Game Settings**
- House edge: **1% – 20%**  
- Tweak methods: Weighted probabilities, modified payouts, or both  
- Color selection: **Red, Green, Blue, Yellow, White, Black**

---

### **3. Comprehensive Visualization Dashboard**
- Balance progression (line charts)
- Win/loss distribution (pie charts)
- Probability analysis (fair vs tweaked)
- Statistical summaries:
  - Mean, median, standard deviation
  - Skewness, kurtosis  
- Performance metrics:
  - Sharpe ratio  
  - Max drawdown  
  - Risk of ruin

---

### **4. Advanced Analytics**
- Monte Carlo Simulation (10,000+ iterations)
- Expected value calculation
- House edge quantification
- Convergence analysis (law of large numbers)
- Bankroll risk assessment and survival analysis

---

### **5. Comparative Analysis**
- Fair vs Tweaked results side-by-side
- Statistical significance testing
- Visualization overlays
- Risk-adjusted performance comparison

---

## 🚀 Installation

### **Clone the repository**
```bash
git clone <repository-url>
cd stochastic-game-simulation
```

## 📦 Install Dependencies **
    pip install -r requirements.txt

### Dependencies
     streamlit==1.31.0
     numpy>=1.26.0
     pandas>=2.2.0
     matplotlib>=3.9.0
     seaborn>=0.13.2
     plotly>=5.17.0
     scipy>=1.11.4
---

## 🎮 Usage

### Run the Streamlit App
    streamlit run perya_game_simulation.py


### For the Deployed App
Open your browser at:
https://jarosales-arch-group1-stochastic-g-perya-game-simulation-cwd6q1.streamlit.app/

---

## ⚙️ App Workflow

### 1.📂 Sidebar Control/Configure Simulation
- Set simulation size
- Set bet amount
- Set starting balance
- Game type selection (Fair / Tweaked / Compare Both)
- House edge & tweak settings
- Select betting color

### 2. Run Simulation
- Click Run Simulation
- Watch real-time progress and dynamic charts

### 3. Analyze Results
- Balance history
- Win/loss distribution
- Probability analysis
- Statistical summary
- Payout Distribution

### 4. Comparative Mode
- Fair vs Tweaked difference
- House edge analysis
- Risk-adjusted return metrics

---

## 📊 App Workflow Summary
    Configuration → Simulation → Visualization → Analysis → Comparison → Insights

---

## 🎓 Educational Purpose
This application helps learners:

- Understand stochastic processes
- Learn Monte Carlo simulation
- Analyze casino mathematics and house edge
- Visualize convergence (law of large numbers)
- Practice statistical interpretation
- Study risk-of-ruin and bankroll theory

---

## 🎨 UI Features
- Clean, responsive layout
- Interactive Plotly graphs
- Organized analysis tabs
- Real-time progress indicators
- Downloadable simulation results
- Professional visual styling

---

## 📝 Mathematical Foundation

### Fair Game
    Probability per color = 1/6 ≈ 16.67%
    Payout multiplier = 5x
    Expected Value (EV) = 0
    House Edge = 0%

### Tweaked Game
    Weighted probabilities (e.g., 14% vs 20%)
    Reduced payout multiplier (e.g., 4.8x)
    Expected Value = negative
    House Edge = (Fair EV - Actual EV) / Bet × 100%

### Key Formulas
    EV = Σ(Probability × Payout) - Cost
    House Edge = 1 - (Player EV / Bet Amount)
    Risk of Ruin = ((1 - p)/p)^(B/b)

---

## 🔍 Example Use Cases
- Learning probability theory
- Understanding house edge
- Testing bankroll strategies
- Running long-term simulations
- Fair vs tweaked model comparison
- Exploring convergence of probabilities

---

## 🛠️ Customization
Possible enhancements:

- Add more game types
- Add betting strategies
- Add time series analysis
- Add more statistical tests
- Multiplayer mode
- Animations and sound effects

---

## 📄 License
This project is open-source and intended for educational use.

---

## 🤝 Contributing
Contributions are welcome! You may:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation
- Add new game models

---

## 💡 Tips for Best Results
- Start with small simulations (1,000 games)
- Use comparison mode for deeper insights
- Adjust only one parameter at a time
- Run 10,000+ games for stable statistics
- Experiment with bet sizes
- Track both financial and statistical metrics

