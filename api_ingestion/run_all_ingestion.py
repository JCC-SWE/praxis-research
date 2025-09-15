import sys
import os
from datetime import datetime
import time
# Add path setup for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'azure_resources'))

# Import all ingestion functions
from arxiv_ingest import batch_get_arxiv_to_cosmos_multi_topic
from event_registry_ingest import fetch_semanticscholar_batch_to_cosmos_multi_topic 
from open_alex import fetch_openalex_batch_to_cosmos
from semantic_scholar import fetch_semanticscholar_batch_to_cosmos_multi_topic

def load_topics_from_file(file_path="ai_topics.txt"):
    """Load topics from a text file, one topic per line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
        return topics
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default topics.")
        return ['machine learning', 'artificial intelligence', 'deep learning']
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Using default topics.")
        return ['machine learning', 'artificial intelligence', 'deep learning']
topics = load_topics_from_file()

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
        
        # 1. Semantic Scholar ingestion using helper function
        print("\n📚 Ingesting from Semantic Scholar...")
        try:
            result = run_semantic_scholar_only(container_name=container_name)
            ingestion_results['results']['semantic_scholar'] = {'status': 'success', 'result': result}
            print("✅ Semantic Scholar ingestion complete")
        except Exception as e:
            ingestion_results['results']['semantic_scholar'] = {'status': 'failed', 'error': str(e)}
            print(f"❌ Semantic Scholar ingestion failed: {e}")
        
        # 2. ArXiv ingestion using helper function
        print("\n📖 Ingesting from ArXiv...")
        try:
            result = run_arxiv_only(
                container_name=container_name,
                date_start=date_start,
                date_end=date_end
            )
            ingestion_results['results']['arxiv'] = {'status': 'success', 'result': result}
            print("✅ ArXiv ingestion complete")
        except Exception as e:
            ingestion_results['results']['arxiv'] = {'status': 'failed', 'error': str(e)}
            print(f"❌ ArXiv ingestion failed: {e}")
        
        # 3. OpenAlex ingestion using helper function
        print("\n🔬 Ingesting from OpenAlex...")
        try:
            result = run_openalex_only(container_name=container_name)
            ingestion_results['results']['openalex'] = {'status': 'success', 'result': result}
            print("✅ OpenAlex ingestion complete")
        except Exception as e:
            ingestion_results['results']['openalex'] = {'status': 'failed', 'error': str(e)}
            print(f"❌ OpenAlex ingestion failed: {e}")
        
        # 4. Event Registry ingestion using helper function
        print("\n📰 Ingesting from Event Registry...")
        try:
            result = run_event_registry_only(
                container_name=container_name,
                date_start=date_start,
                date_end=date_end
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

def run_semantic_scholar_only(container_name="baseline-papers"):
    """Run only Semantic Scholar ingestion."""
    print("📚 Running Semantic Scholar ingestion only...")
    return fetch_semanticscholar_batch_to_cosmos_multi_topic(
        topics=topics,
        limit=500,
        year_range=('01-2024', '09-2024'),
        container_name=container_name
    )

def run_arxiv_only(container_name="s_scholar_container", date_start="2025-01-01", date_end="2025-07-31"):
    """Run only ArXiv ingestion."""
    print("📖 Running ArXiv ingestion only...")
    return batch_get_arxiv_to_cosmos_multi_topic(
        search_query=topics,
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
        search_terms=topics,
        total_papers=1000,
        batch_size=200,
        year_range=('01-2024', '09-2024'),
        container_name=container_name
    )

def batch_topics(topics, batch_size=15):
    """Split topics into batches of specified size."""
    for i in range(0, len(topics), batch_size):
        yield topics[i:i + batch_size]

def run_event_registry_only(container_name="s_scholar_container", date_start="2025-01-01", date_end="2025-07-31"):
    """Run only Event Registry ingestion with batched topics."""
    print("📰 Running Event Registry ingestion only...")
    
    results = []
    total_batches = (len(topics) + 14) // 15  # Calculate number of batches
    
    for batch_num, topic_batch in enumerate(batch_topics(topics, 15), 1):
        print(f"Processing batch {batch_num}/{total_batches} with {len(topic_batch)} topics...")
        
        try:
            result = fetch_eventregistry_to_cosmos(
                keywords=topic_batch,
                max_items=500,
                date_range=(date_start, date_end),
                container_name=container_name
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            continue
    
    print(f"Completed processing {len(results)} batches")
    return results

if __name__ == "__main__":
    # Choose what to run:
    
    # Option 1: Run all ingestion (takes significant time!)
    results = run_all_ingestion(container_name="s_scholar_container",date_start="2024-01-01", date_end="2024-09-39",)  # Run 3 times
    
    # Option 2: Run individual sources (uncomment as needed)
    #run_semantic_scholar_only()
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