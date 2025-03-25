"""
Example script demonstrating the multi-agent trading system.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
import json

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import DataManager
from src.agent.multi_agent import TradingAgentOrchestrator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-agent trading system demo')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to analyze')
    
    parser.add_argument('--days', type=int, default=90,
                       help='Number of days to analyze')
    
    parser.add_argument('--request', type=str, 
                       default='Analyze the recent market trends and provide a trading recommendation',
                       help='Analysis request for the agent system')
    
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key (optional)')
    
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0: silent, 1: normal, 2: detailed)')
    
    return parser.parse_args()

def main():
    """Main function to run the example"""
    args = parse_args()
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # Initialize the data manager
    data_manager = DataManager(
        market_data_source="yahoo",
        news_data_source="news",  # This will use our simulated news data
        verbose=args.verbose
    )
    
    # Initialize the orchestrator
    orchestrator = TradingAgentOrchestrator(
        data_manager=data_manager,
        openai_api_key=args.api_key,
        verbose=args.verbose
    )
    
    # Process the request
    print(f"\n=== Analyzing {args.symbol} from {start_date} to {end_date} ===\n")
    
    result = orchestrator.process_request(
        request=args.request,
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print the results
    print("\n=== Analysis Results ===\n")
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Symbol: {result['symbol']}")
        print(f"Date Range: {result['date_range']['start_date']} to {result['date_range']['end_date']}")
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {result['explanation']}")
        
        print("\nRecommended Actions:")
        for action in result.get('recommended_actions', []):
            print(f"- {action['action'].upper()} {action['symbol']}: {action['reason']}")
        
        print("\nDetailed Analysis:")
        print(result['analysis'])
    
    # Save the results to a JSON file
    output_file = f"results/multi_agent_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 