"""
Interactive Chat Interface for Valio AI

Run this and ask questions in natural language!

Examples:
- "What replacements do we have for product 6409460002724?"
- "Show me shortage events from sales data"
- "What's the forecast for next week?"
- "How many products are in the catalog?"
"""

from backend.conversational_agent import ConversationalAgent

def main():
    print("="*70)
    print("  VALIO AI - CONVERSATIONAL SUPPLY CHAIN ASSISTANT")
    print("="*70)
    print()
    print("Ask me anything about:")
    print("  - Product replacements")
    print("  - Demand forecasting")
    print("  - Shortage detection")
    print("  - Sales data analysis")
    print()
    print("Type 'exit' or 'quit' to end the conversation")
    print("="*70)
    print()

    # Initialize agent
    print("Initializing AI agent...")
    try:
        agent = ConversationalAgent()
        print("Ready! Connected to LM Studio.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure LM Studio is running on http://localhost:1234")
        return

    print()

    # Chat loop
    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\nGoodbye!")
            break

        # Get response
        print("\nValio AI: Thinking...")
        print()

        try:
            response = agent.answer_question(user_input)

            print("\n" + "="*70)
            print("Valio AI:")
            print("="*70)
            print(response['answer'])

            # Optionally show generated code
            if response.get('code_generated'):
                show_code = input("\n[Show generated code? (y/n)]: ").strip().lower()
                if show_code == 'y':
                    print("\nGenerated Python Code:")
                    print("-"*70)
                    print(response['code_generated'])

        except Exception as e:
            print(f"\nError: {e}")
            print("Try rephrasing your question or check LM Studio connection.")


if __name__ == "__main__":
    main()
