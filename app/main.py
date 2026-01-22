"""
TradeSense - Multi-Factor Trading Signal Scanner

A sophisticated trading application that combines technical analysis,
fundamental data, sentiment analysis, and options flow to generate
high-confidence buy/sell signals with take profit and stop loss targets.
"""

import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="TradeSense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .signal-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .buy-signal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
        border-color: #10b981;
    }
    .sell-signal {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1));
        border-color: #ef4444;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


def show_dashboard():
    """Main dashboard page."""
    st.markdown('<h1 class="main-header">📈 TradeSense</h1>', unsafe_allow_html=True)
    st.markdown("### Multi-Factor Trading Signal Scanner")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        asset_type = st.selectbox(
            "Asset Category",
            ["All", "Stocks", "ETFs", "Crypto"]
        )
        
        confidence_threshold = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        st.markdown("---")
        st.markdown("### 📊 Data Sources")
        include_technical = st.checkbox("Technical Analysis", value=True)
        include_fundamental = st.checkbox("Fundamental Analysis", value=True)
        include_sentiment = st.checkbox("Sentiment Analysis", value=True)
        include_options = st.checkbox("Options Flow", value=True)
        
        st.markdown("---")
        scan_button = st.button("🔍 Scan Watchlist", type="primary", use_container_width=True)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Buy Signals", "0", delta=None)
    with col2:
        st.metric("📉 Sell Signals", "0", delta=None)
    with col3:
        st.metric("⏸️ Hold", "20", delta=None)
    with col4:
        st.metric("🎯 Avg Confidence", "0.0%", delta=None)
    
    st.markdown("---")
    
    # Single ticker analysis
    st.markdown("### 🔎 Analyze Single Ticker")
    
    ticker_col, analyze_col = st.columns([3, 1])
    with ticker_col:
        ticker_input = st.text_input("Enter ticker symbol", placeholder="AAPL, MSFT, BTC-CAD...")
    with analyze_col:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("Analyze", type="secondary", use_container_width=True)
    
    if analyze_button and ticker_input:
        with st.spinner(f"Analyzing {ticker_input.upper()}..."):
            try:
                from src.signals import signal_generator
                
                signal = signal_generator.generate_signal(
                    ticker_input.upper(),
                    include_fundamentals=include_fundamental,
                    include_sentiment=include_sentiment,
                    include_options=include_options
                )
                
                # Display signal card
                signal_class = ""
                if signal.signal_type.value == "BUY":
                    signal_class = "buy-signal"
                    signal_emoji = "🟢"
                elif signal.signal_type.value == "SELL":
                    signal_class = "sell-signal"
                    signal_emoji = "🔴"
                else:
                    signal_emoji = "⚪"
                
                st.markdown(f"""
                <div class="signal-card {signal_class}">
                    <h2>{signal_emoji} {signal.ticker} - {signal.signal_type.value}</h2>
                    <p><strong>Confidence:</strong> {signal.confidence:.0%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entry Price", f"${signal.entry_price:.2f}")
                with col2:
                    tp_val = f"${signal.take_profit:.2f}" if signal.take_profit else "N/A"
                    st.metric("Take Profit", tp_val)
                with col3:
                    sl_val = f"${signal.stop_loss:.2f}" if signal.stop_loss else "N/A"
                    st.metric("Stop Loss", sl_val)
                with col4:
                    st.metric("Confidence", f"{signal.confidence:.0%}")
                
                # Score breakdown
                st.markdown("### 📊 Score Breakdown")
                score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                
                with score_col1:
                    st.progress(signal.technical_score)
                    st.caption(f"Technical: {signal.technical_score:.0%}")
                with score_col2:
                    st.progress(signal.fundamental_score)
                    st.caption(f"Fundamental: {signal.fundamental_score:.0%}")
                with score_col3:
                    st.progress(signal.sentiment_score)
                    st.caption(f"Sentiment: {signal.sentiment_score:.0%}")
                with score_col4:
                    st.progress(signal.options_score)
                    st.caption(f"Options: {signal.options_score:.0%}")
                
                # Reasons
                if signal.reasons:
                    st.markdown("### 💡 Signal Reasons")
                    for reason in signal.reasons:
                        st.markdown(f"- {reason}")
                
            except Exception as e:
                st.error(f"Error analyzing {ticker_input}: {str(e)}")
    
    # Recent signals placeholder
    st.markdown("---")
    st.markdown("### 📋 Recent Signals")
    
    st.info("Click 'Scan Watchlist' to generate signals for all tracked assets.")
    
    if scan_button:
        with st.spinner("Scanning watchlist..."):
            try:
                from src.signals import signal_generator, SignalType
                
                category = asset_type.lower() if asset_type != "All" else "all"
                signals = signal_generator.scan_watchlist(category=category)
                
                # Filter by confidence
                filtered = [s for s in signals if s.confidence >= confidence_threshold]
                
                # Display summary
                buy_count = sum(1 for s in filtered if s.signal_type == SignalType.BUY)
                sell_count = sum(1 for s in filtered if s.signal_type == SignalType.SELL)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Buy Signals", buy_count)
                with col2:
                    st.metric("Sell Signals", sell_count)
                with col3:
                    avg_conf = sum(s.confidence for s in filtered) / len(filtered) if filtered else 0
                    st.metric("Avg Confidence", f"{avg_conf:.0%}")
                
                # Display table
                if filtered:
                    import pandas as pd
                    
                    df = pd.DataFrame([
                        {
                            "Ticker": s.ticker,
                            "Signal": s.signal_type.value,
                            "Confidence": f"{s.confidence:.0%}",
                            "Entry": f"${s.entry_price:.2f}",
                            "Take Profit": f"${s.take_profit:.2f}" if s.take_profit else "N/A",
                            "Stop Loss": f"${s.stop_loss:.2f}" if s.stop_loss else "N/A",
                        }
                        for s in filtered
                    ])
                    
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No signals meeting confidence threshold.")
                    
            except Exception as e:
                st.error(f"Error scanning watchlist: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<center><small>TradeSense v0.1.0 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small></center>",
        unsafe_allow_html=True
    )


def main():
    """Main application entry point."""
    show_dashboard()


if __name__ == "__main__":
    main()
