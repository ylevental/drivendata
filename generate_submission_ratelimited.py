#!/usr/bin/env python3
"""DrivenData Academic Summarization - Rate Limited Version"""

import pandas as pd
import anthropic
import os
import json
import time
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent

def create_summarization_prompt(text: str) -> str:
    return f"""Write a concise 180-200 word abstract for this research paper.

CRITICAL REQUIREMENTS:
- Length: EXACTLY 180-200 words (count carefully!)
- Start with: "This study examines..." or "This article explores..." or similar
- Use phrases directly from the paper when describing methods and findings
- Be direct and straightforward - no flowery language

Structure:
- Opening: Research question/gap (2-3 sentences)
- Middle: Methods and approach (2-3 sentences)  
- End: Key findings and contributions (2-3 sentences)

PAPER:
{text}

ABSTRACT (180-200 words):"""

def generate_summary(text: str, client: anthropic.Anthropic) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": create_summarization_prompt(text)}]
            )
            return message.content[0].text.strip()
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)
                print(f"\n⏳ Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'summaries': []}

def save_progress(progress_file: Path, progress: dict):
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def main():
    print("="*70)
    print(" DrivenData Academic Summarization - Rate Limited")
    print("="*70)
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ ANTHROPIC_API_KEY not found")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")
    progress_file = DATA_DIR / "progress.json"
    progress = load_progress(progress_file)
    completed_ids = set(progress['completed'])
    summaries = {s['paper_id']: s['summary'] for s in progress['summaries']}
    
    remaining = len(test_df) - len(completed_ids)
    print(f"\n✅ Already completed: {len(completed_ids)}/345")
    print(f"📝 Remaining: {remaining} papers")
    print(f"⏱️  Estimated time: ~{remaining * 25 / 60:.0f} minutes")
    
    
    print(f"\n🤖 Generating with 15s delays...")
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        paper_id = int(row['paper_id'])
        
        if paper_id in completed_ids:
            continue
        
        try:
            summary = generate_summary(row['text'], client)
            summaries[paper_id] = summary
            
            progress['completed'].append(paper_id)
            progress['summaries'].append({'paper_id': paper_id, 'summary': summary})
            
            if len(progress['completed']) % 10 == 0:
                save_progress(progress_file, progress)
            
            time.sleep(15)  # Rate limit: 2.4 papers/min
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            save_progress(progress_file, progress)
            return
    
    save_progress(progress_file, progress)
    
    submission_data = [{'paper_id': pid, 'summary': summaries[pid]} for pid in sorted(summaries.keys())]
    submission_df = pd.DataFrame(submission_data)
    submission_path = DATA_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✅ Complete! {len(summaries)}/345")
    print(f"📁 Saved to: {submission_path}")

if __name__ == "__main__":
    main()
