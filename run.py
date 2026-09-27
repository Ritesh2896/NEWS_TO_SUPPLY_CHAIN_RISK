import os, subprocess, sys
from dotenv import load_dotenv
load_dotenv()

def main():
    topic=os.getenv('LIVE_NEWS_TOPIC','supply chain disruptions ports logistics manufacturing trade')
    provider=os.getenv('LIVE_NEWS_PROVIDER','both')
    hours=int(os.getenv('LIVE_NEWS_HOURS','48'))
    limit=int(os.getenv('LIVE_NEWS_LIMIT','8'))
    from src.ingestion.live_news import search_live_news
    from src.pipeline.end_to_end_pipeline import run_pipeline
    print('Searching live web...')
    items=search_live_news(topic,provider,hours,limit)
    import pandas as pd
    os.makedirs('data/processed',exist_ok=True)
    pd.DataFrame(items).to_csv('data/processed/live_news.csv',index=False)
    result=run_pipeline(items)
    print(f'Retrieved {len(items)} live articles; generated {len(result)} alerts.')
    subprocess.run(['streamlit','run','dashboard/app.py'],check=True)
if __name__=='__main__': main()
