"""
Test ML-integrated SignalGenerator
"""
from src.signals import signal_generator

def test_ml_signal():
    print("=" * 50)
    print("Testing ML-Integrated Signal Generator")
    print("=" * 50)
    
    # Test with a trained model available
    signal = signal_generator.generate_signal("AAPL")
    
    print(f"\nTicker: {signal.ticker}")
    print(f"Signal: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"\nComponent Scores:")
    print(f"  Technical:   {signal.technical_score:.2%}")
    print(f"  Fundamental: {signal.fundamental_score:.2%}")
    print(f"  Sentiment:   {signal.sentiment_score:.2%}")
    print(f"  Options:     {signal.options_score:.2%}")
    print(f"  ML Model:    {signal.ml_score:.2%}")
    
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    
    # Check if ML reason is present
    ml_reasons = [r for r in signal.reasons if "ML Model" in r]
    if ml_reasons:
        print("\n✅ ML Model integrated successfully!")
    else:
        print("\n⚠️ ML Model not contributing (model may not be trained)")

if __name__ == "__main__":
    test_ml_signal()
