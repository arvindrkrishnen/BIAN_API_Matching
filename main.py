# Run this code on Colab

import os
import yaml
import requests
import zipfile
import pandas as pd
from io import BytesIO
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def get_repo_info(repo_url: str) -> Tuple[str, str]:
    """
    Get repository information from GitHub API.
    """
    # Extract owner and repo name from the URL
    parts = repo_url.split('/')
    owner, repo = parts[-2], parts[-1]
    
    # Make a request to the GitHub API
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        data = response.json()
        return data['name'], data['description']
    else:
        print(f"Failed to fetch repository information: HTTP {response.status_code}")
        return repo, "Description not available"

def download_github_repo(repo_url: str, branch: str, save_path: str) -> None:
    """
    Download a GitHub repository as a zip file and extract it.
    """
    zip_url = f"{repo_url}/archive/{branch}.zip"
    response = requests.get(zip_url)
    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(save_path)
        print(f"Repository downloaded and extracted to {save_path}")
    else:
        raise Exception(f"Failed to download repository: HTTP {response.status_code}")

def extract_api_info(file_path: str) -> Dict[str, str]:
    """
    Extract title and description from a YAML file.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return {
                'file': os.path.basename(file_path),
                'api_title': data.get('info', {}).get('title', 'No title found'),
                'api_description': data.get('info', {}).get('description', 'No description found')
            }
        except yaml.YAMLError as e:
            print(f"Error parsing {file_path}: {e}")
            return None

def process_yaml_files(directory: str) -> List[Dict[str, str]]:
    """
    Process all YAML files in the given directory and its subdirectories.
    """
    api_info_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                file_path = os.path.join(root, file)
                api_info = extract_api_info(file_path)
                if api_info:
                    api_info_list.append(api_info)
    return api_info_list

def download_bian_api():
    # GitHub repository details
    repo_url = "https://github.com/bian-official/public"
    branch = "main"
    
    # Get repository information
    repo_name, repo_description = get_repo_info(repo_url)
    print(f"Repository: {repo_name}")
    print(f"Description: {repo_description}")
    print("-" * 50)
    
    # Local path to save and process files
    local_path = "/content/yaml"
    
    # Download and extract the repository
    download_github_repo(repo_url, branch, local_path)
    
    # Path to the YAML files within the extracted repository
    yaml_path = os.path.join(local_path, "public-main", "release12.0.0", "semantic-apis", "oas3", "yamls")
    
    # Process the YAML files
    api_info_list = process_yaml_files(yaml_path)
    
    # Add repository information to each API info dictionary
    for api_info in api_info_list:
        api_info['repo_name'] = repo_name
        api_info['repo_description'] = repo_description
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(api_info_list)
    
    # Reorder columns to put repo information first
    df = df[['repo_name', 'repo_description', 'file', 'api_title', 'api_description']]
    
    # Optionally, save the DataFrame to a CSV file
    df.to_csv('api_info.csv', index=False)
    
    return df





def find_best_match(query: str, df: pd.DataFrame) -> Tuple[str, float]:
    """
    Find the best matching api_title for a given query description.
    
    Args:
    query (str): The description to match against.
    df (pd.DataFrame): The DataFrame containing 'api_description' and 'api_title'.
    
    Returns:
    Tuple[str, float]: The best matching api_title and its similarity score.
    """
    # Combine the query with all descriptions
    all_descriptions = df['api_description'].tolist() + [query]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Find the index of the best match
    best_match_index = cosine_similarities.argmax()
    
    return df.iloc[best_match_index]['api_title'], df.iloc[best_match_index]['api_description'],cosine_similarities[best_match_index]


if __name__ == "__main__":
    result_df = download_bian_api()
    #print(result_df)

    # Example usage of find_best_match function
    query_description = "internal audit checks"
    best_match_title, api_description, similarity_score = find_best_match(query_description, result_df)
    print(f"\nBest match for '{query_description}':")
    print(f"API Title: {best_match_title}")
    print(f"API Description: {api_description}")
    print(f"Similarity Score: {similarity_score:.2f}")

