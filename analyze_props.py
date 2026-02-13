#!/usr/bin/env python3
"""
Quick PrizePicks Analysis CLI Tool.
Analyze props from command line or file input.

Usage:
    # Analyze from command line
    python analyze_props.py "LeBron James, points, 25.5" "Stephen Curry, threes, 4.5"

    # Analyze from file
    python analyze_props.py -f props.txt

    # Interactive mode
    python analyze_props.py -i
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prizepicks import PrizePicksAnalyzer, quick_analyze


def parse_prop(prop_str: str) -> dict:
    """Parse a prop string into a dict."""
    parts = [p.strip() for p in prop_str.split(',')]
    if len(parts) < 3:
        raise ValueError(f"Invalid prop format: {prop_str}")
    return {
        'player': parts[0],
        'prop_type': parts[1].lower(),
        'line': float(parts[2])
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze PrizePicks props against model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "LeBron James, points, 25.5"
  %(prog)s "Curry, threes, 4.5" "Jokic, rebounds, 12.5"
  %(prog)s -f props.txt
  %(prog)s -i

Prop format: "Player Name, prop_type, line"
Supported prop types: points, rebounds, assists, steals, blocks, threes, turnovers
        """
    )

    parser.add_argument(
        'props',
        nargs='*',
        help='Props to analyze in format "Player, prop_type, line"'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Read props from file (one per line)'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive mode - enter props one at a time'
    )
    parser.add_argument(
        '--min-edge',
        type=float,
        default=5.0,
        help='Minimum edge %% to highlight (default: 5)'
    )
    parser.add_argument(
        '--parlay',
        type=int,
        default=0,
        help='Build optimal parlay with N legs'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Initialize analyzer
    print("Loading models...")
    analyzer = PrizePicksAnalyzer()
    analyzer.load_models()

    if not analyzer._models_loaded:
        print("ERROR: Models not loaded. Run 'python train_models.py' first.")
        sys.exit(1)

    props = []

    # Interactive mode
    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Enter props in format: Player Name, prop_type, line")
        print("Type 'done' when finished, 'quit' to exit\n")

        while True:
            try:
                line = input("Enter prop: ").strip()
                if line.lower() == 'quit':
                    sys.exit(0)
                if line.lower() == 'done':
                    break
                if line:
                    props.append(parse_prop(line))
                    print(f"  Added: {props[-1]['player']} {props[-1]['prop_type']} {props[-1]['line']}")
            except ValueError as e:
                print(f"  Error: {e}")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)

    # Read from file
    elif args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)

        with open(args.file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        props.append(parse_prop(line))
                    except ValueError as e:
                        print(f"Warning: {e}")

    # Command line props
    elif args.props:
        for prop_str in args.props:
            try:
                props.append(parse_prop(prop_str))
            except ValueError as e:
                print(f"Warning: {e}")

    else:
        parser.print_help()
        sys.exit(0)

    if not props:
        print("No valid props to analyze.")
        sys.exit(1)

    print(f"\nAnalyzing {len(props)} props...\n")

    # Analyze
    picks = analyzer.analyze_slate(props)

    if not picks:
        print("Could not analyze any props. Check player names.")
        sys.exit(1)

    # Output as JSON
    if args.json:
        import json
        output = []
        for pick in picks:
            output.append({
                'player': pick.player_name,
                'prop_type': pick.prop_type,
                'line': pick.line,
                'prediction': round(pick.prediction, 1),
                'diff': round(pick.diff_from_line, 1),
                'edge': round(pick.edge_pct, 1),
                'recommendation': pick.recommendation,
                'confidence': pick.confidence,
                'probability': round(pick.probability * 100, 1),
                'hit_rate_l10': round(pick.recent_hit_rate * 100, 0),
                'trend': pick.trend,
                'minutes_concern': pick.minutes_concern
            })
        print(json.dumps(output, indent=2))
        return

    # Print table
    print(analyzer.format_picks_table(picks))

    # Highlight best picks
    min_edge = args.min_edge / 100
    best = [p for p in picks if p.edge >= min_edge and p.confidence in ['strong', 'moderate']]

    if best:
        print(f"\n🏆 TOP PICKS (>={args.min_edge}% edge):\n")
        for i, pick in enumerate(best[:5], 1):
            trend_emoji = "🔥" if pick.trend == 'hot' else "❄️" if pick.trend == 'cold' else "➡️"
            print(f"  {i}. {pick.player_name} {pick.recommendation} {pick.line} {pick.prop_type}")
            print(f"     Prediction: {pick.prediction:.1f} | Edge: {pick.edge_pct:.1f}% | {pick.confidence.upper()} {trend_emoji}")
            if pick.minutes_concern:
                print(f"     ⚠️  Minutes concern")
            print()

    # Build parlay if requested
    if args.parlay > 0:
        parlay_picks, parlay_prob = analyzer.build_parlay(props, legs=args.parlay)

        print(f"\n🎲 SUGGESTED {args.parlay}-LEG PARLAY:")
        print(f"   Combined Probability: {parlay_prob*100:.1f}%\n")

        for i, pick in enumerate(parlay_picks, 1):
            print(f"   Leg {i}: {pick.player_name} {pick.recommendation} {pick.line} {pick.prop_type} (Edge: {pick.edge_pct:.1f}%)")

    print()


if __name__ == '__main__':
    main()
