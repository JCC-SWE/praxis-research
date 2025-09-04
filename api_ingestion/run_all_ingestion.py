import sys
import os
from datetime import datetime

# Add path setup for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'azure_resources'))

# Import all ingestion functions
from arxiv_ingest import batch_get_arxiv_to_cosmos
from event_registry_ingest import fetch_eventregistry_to_cosmos  
from open_alex import fetch_openalex_batch_to_cosmos
from semantic_scholar import fetch_semanticscholar_batch_to_cosmos

def run_all_ingestion(container_name="s_scholar_container", 
                     date_start="2025-01-01", 
                     date_end="2025-07-31",
                     iterations=1):
    """
    Run all data ingestion functions to populate CosmosDB container.
    
    Args:
        container_name (str): Target CosmosDB container
        date_start (str): Start date for data collection
        date_end (str): End date for data collection
        iterations (int): Number of times to run the full ingestion cycle
    """
    print(f"🚀 Starting Complete Data Ingestion Pipeline ({iterations} iterations)")
    print(f"Target container: {container_name}")
    print(f"Date range: {date_start} to {date_end}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for iteration in range(iterations):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {iteration + 1} of {iterations}")
        print(f"{'='*60}")
        
        ingestion_results = {
            'iteration': iteration + 1,
            'start_time': datetime.now().isoformat(),
            'container': container_name,
            'date_range': (date_start, date_end),
            'results': {}
        }
        
        # 1. Semantic Scholar ingestion
        print("\n📚 Ingesting from Semantic Scholar...")
        try:
            result = fetch_semanticscholar_batch_to_cosmos(
                topics=['machine learning', 'artificial intelligence', 'deep learning'],
                limit=500,
                year_range=('01-2025', '07-2025'),
                container_name=container_name
            )
            ingestion_results['results']['semantic_scholar'] = {'status': 'success', 'result': result}
            print("✅ Semantic Scholar ingestion complete")
        except Exception as e:
            ingestion_results['results']['semantic_scholar'] = {'status': 'failed', 'error': str(e)}
            print(f"❌ Semantic Scholar ingestion failed: {e}")
        
        # 2. ArXiv ingestion
        print("\n📖 Ingesting from ArXiv...")
        try:
            result = batch_get_arxiv_to_cosmos(
                search_query="machine learning",
                total_limit=1000,
                batch_size=100,
                container_name=container_name,
                start_date=date_start,
                end_date=date_end
            )
            ingestion_results['results']['arxiv'] = {'status': 'success', 'result': result}
            print("✅ ArXiv ingestion complete")
        except Exception as e:
            ingestion_results['results']['arxiv'] = {'status': 'failed', 'error': str(e)}
            print(f"❌ ArXiv ingestion failed: {e}")
        
        # 3. OpenAlex ingestion
        print("\n🔬 Ingesting from OpenAlex...")
        try:
            result = fetch_openalex_batch_to_cosmos(
                search_terms=["machine learning", "artificial intelligence"],
                total_papers=1000,
                batch_size=200,
                year_range=('01-2025', '07-2025'),
                container_name=container_name
            )
            ingestion_results['results']['openalex'] = {'status': 'success', 'result': result}
            print("✅ OpenAlex ingestion complete")
        except Exception as e:
            ingestion_results['results']['openalex'] = {'status': 'failed', 'error': str(e)}
            print(f"❌ OpenAlex ingestion failed: {e}")
        
        # 4. Event Registry ingestion
        print("\n📰 Ingesting from Event Registry...")
        try:
            result = fetch_eventregistry_to_cosmos(
                keywords=["artificial intelligence", "machine learning", "deep learning"],
                max_items=500,
                date_range=(date_start, date_end),
                container_name=container_name
            )
            ingestion_results['results']['event_registry'] = {'status': 'success', 'result': result}
            print("✅ Event Registry ingestion complete")
        except Exception as e:
            ingestion_results['results']['event_registry'] = {'status': 'failed', 'error': str(e)}
            print(f"❌ Event Registry ingestion failed: {e}")
        
        # Iteration summary
        ingestion_results['end_time'] = datetime.now().isoformat()
        all_results.append(ingestion_results)
        
        successful = sum(1 for r in ingestion_results['results'].values() if r['status'] == 'success')
        total = len(ingestion_results['results'])
        
        print(f"\n✅ Iteration {iteration + 1} Complete: {successful}/{total} sources successful")
        
        # Short break between iterations (except last one)
        if iteration < iterations - 1:
            print("⏳ Waiting 30 seconds before next iteration...")
            import time
            time.sleep(30)
    
    # Final summary
    print(f"\n🎉 All {iterations} Iterations Complete!")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall summary
    total_successes = 0
    total_attempts = 0
    
    for result in all_results:
        for source, outcome in result['results'].items():
            total_attempts += 1
            if outcome['status'] == 'success':
                total_successes += 1
    
    print(f"\n📊 Overall Summary:")
    print(f"   Total attempts: {total_attempts}")
    print(f"   Total successes: {total_successes}")
    print(f"   Success rate: {total_successes/total_attempts*100:.1f}%")
    
    return all_results

def run_semantic_scholar_only(container_name="s_scholar_container"):
    """Run only Semantic Scholar ingestion."""
    print("📚 Running Semantic Scholar ingestion only...")
    return fetch_semanticscholar_batch_to_cosmos(
        topics=['machine learning', 'artificial intelligence', 'deep learning'],
        limit=500,
        year_range=('01-2025', '07-2025'),
        container_name=container_name
    )

def run_arxiv_only(container_name="s_scholar_container", date_start="2025-01-01", date_end="2025-07-31"):
    """Run only ArXiv ingestion."""
    print("📖 Running ArXiv ingestion only...")
    return batch_get_arxiv_to_cosmos(
        search_query="machine learning",
        total_limit=1000,
        batch_size=100,
        container_name=container_name,
        start_date=date_start,
        end_date=date_end
    )

def run_openalex_only(container_name="s_scholar_container"):
    """Run only OpenAlex ingestion."""
    print("🔬 Running OpenAlex ingestion only...")
    return fetch_openalex_batch_to_cosmos(
        search_terms=["machine learning", "artificial intelligence"],
        total_papers=1000,
        batch_size=200,
        year_range=('01-2025', '07-2025'),
        container_name=container_name
    )

def run_event_registry_only(container_name="s_scholar_container", date_start="2025-01-01", date_end="2025-07-31"):
    """Run only Event Registry ingestion."""
    print("📰 Running Event Registry ingestion only...")
    return fetch_eventregistry_to_cosmos(
        keywords=["artificial intelligence", "machine learning", "deep learning"],
        max_items=500,
        date_range=(date_start, date_end),
        container_name=container_name
    )

if __name__ == "__main__":
    # Choose what to run:
    
    # Option 1: Run all ingestion (takes significant time!)
    #results = run_all_ingestion(iterations=3)  # Run 3 times
    
    # Option 2: Run individual sources (uncomment as needed)
    run_semantic_scholar_only()
    # run_arxiv_only()
    # run_openalex_only()
    # run_event_registry_only()
    
    # Option 3: Custom parameters with multiple iterations
    # results = run_all_ingestion(
    #     container_name="s_scholar_container",
    #     date_start="2025-06-01", 
    #     date_end="2025-07-31",
    #     iterations=5
    # )