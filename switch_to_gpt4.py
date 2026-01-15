#!/usr/bin/env python3
"""
Quick script to switch models between GPT-4 and GPT-4o-mini

Usage:
    python3 switch_to_gpt4.py          # Switch to GPT-4
    python3 switch_to_gpt4.py mini     # Switch to GPT-4o-mini
"""

import sys
import re

def update_model(file_path, old_model, new_model):
    """Update model in a file"""
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace the model
    if old_model in content:
        content = content.replace(
            f'llm_model: str = "{old_model}"',
            f'llm_model: str = "{new_model}"'
        )
        content = content.replace(
            f'llm_model="{old_model}"',
            f'llm_model="{new_model}"'
        )

        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    # Determine which model to switch to
    use_mini = len(sys.argv) > 1 and sys.argv[1] == 'mini'

    if use_mini:
        old_model = "gpt-4"
        new_model = "gpt-4o-mini"
        print("🔄 Switching to GPT-4o-mini (cheaper, lower quality)...")
    else:
        old_model = "gpt-4o-mini"
        new_model = "gpt-4"
        print("🔄 Switching to GPT-4 (expensive, higher quality)...")

    files = [
        'src/agents/manager_agent.py',
        'src/agents/summarization_agent.py',
        'src/indexing/build_indexes.py'
    ]

    updated_count = 0
    for file_path in files:
        try:
            if update_model(file_path, old_model, new_model):
                print(f"  ✓ Updated {file_path}")
                updated_count += 1
            else:
                print(f"  ⚠ No changes needed in {file_path}")
        except Exception as e:
            print(f"  ✗ Error updating {file_path}: {e}")

    print(f"\n{'='*60}")
    if updated_count > 0:
        print(f"✅ Successfully updated {updated_count} files to use {new_model}")
        print(f"\n⚠️  IMPORTANT: Restart your Streamlit app to use the new model!")
        print(f"   Run: streamlit run streamlit_app.py")
    else:
        print(f"ℹ️  Already using {new_model} - no changes made")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
