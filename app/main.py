"""
TradeSense - Streamlit Main Application

Entry point for the trading signal scanner dashboard.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="TradeSense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.title("📈 TradeSense")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "Signal Details", "Sentiment", "Backtesting", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("**Quick Stats**")
        st.metric("Active Signals", "0", "0")
        st.metric("Watchlist", "20 tickers")
        st.metric("Last Scan", "N/A")
    
    # Main content
    if page == "Dashboard":
        show_dashboard()
    elif page == "Signal Details":
        show_signal_details()
    elif page == "Sentiment":
        show_sentiment()
    elif page == "Backtesting":
        show_backtesting()
    elif page == "Settings":
        show_settings()


def show_dashboard():
    """Display the main dashboard with active signals."""
    st.title("📊 Dashboard")
    st.markdown("Real-time trading signal scanner")
    
    # Placeholder for development
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BUY Signals", "0", "0", delta_color="normal")
    with col2:
        st.metric("SELL Signals", "0", "0", delta_color="inverse")
    with col3:
        st.metric("Confidence Avg", "N/A")
    with col4:
        st.metric("Last Update", "N/A")
    
    st.markdown("---")
    
    st.info(
        "🚧 **Development in Progress**\n\n"
        "The scanner is being built. Check back after Phase 2 (Data Layer) is complete."
    )


def show_signal_details():
    """Display detailed signal analysis."""
    st.title("🔍 Signal Details")
    st.info("Select a signal from the Dashboard to view details.")


def show_sentiment():
    """Display sentiment analysis dashboard."""
    st.title("🧠 Sentiment Analysis")
    st.info("Sentiment engine will be available after Phase 3.")


def show_backtesting():
    """Display backtesting results."""
    st.title("📈 Backtesting")
    st.info("Backtesting will be available after Phase 9.")


def show_settings():
    """Display settings and configuration."""
    st.title("⚙️ Settings")
    
    st.subheader("Signal Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("RSI Oversold", min_value=0, max_value=100, value=20)
        st.number_input("RSI Overbought", min_value=0, max_value=100, value=80)
    with col2:
        st.number_input("Put/Call Ratio Threshold", min_value=0.0, max_value=2.0, value=0.7)
        st.number_input("Volume Multiplier", min_value=0.1, max_value=5.0, value=1.0)
    
    st.subheader("Scanner Mode")
    st.radio("Mode", ["Real-time", "Batch"], horizontal=True)
    
    st.subheader("Notifications")
    st.checkbox("Enable in-app notifications", value=True)


if __name__ == "__main__":
    main()
